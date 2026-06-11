import os
import math
import random
import logging
import torch
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from nexus.models.policy import load_policy_and_ref_models
from nexus.models.reward import RewardModelScorer, RuleBasedRewardScorer
from nexus.rl.advantages import compute_lpro_advantages
from nexus.rl.loss import compute_dapo_token_loss
from nexus.data.builder import MathDatasetBuilder

log = logging.getLogger(__name__)

class NexusTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if cfg.bf16 and torch.cuda.is_bf16_supported() else torch.float32
        
        self._set_seed()
        self._setup_components()

    def _set_seed(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def _setup_components(self):
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Models
        self.model, self.ref_model = load_policy_and_ref_models(self.cfg.model_name, self.dtype)
        if getattr(self.cfg, "use_lora", False):
            log.info("Finetune LoRA: Đang cấu hình PEFT/LoRA adapter...")
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=self.cfg.lora_target_modules
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        else:
            log.info("Full Finetune: Đang cấu hình để fine-tune toàn bộ model...")

        # Setup Reward Scorer
        if getattr(self.cfg, "use_rule_based_rm", True):
            self.rm_scorer = RuleBasedRewardScorer(self.device)
        else:
            self.rm_scorer = RewardModelScorer(self.cfg.rm_model_name, self.dtype, self.device)

    @staticmethod
    def get_resp_log_probs(model, full_ids: torch.Tensor, prompt_len: int, no_grad: bool = False) -> torch.Tensor:
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            out = model(input_ids=full_ids)
            lp = torch.log_softmax(out.logits[0], dim=-1) # (seq_len, vocab_size)
            resp_ids = full_ids[0, prompt_len:]
            resp_lp = lp[prompt_len - 1 : full_ids.shape[1] - 1] # (resp_len, vocab_size)
            return resp_lp.gather(1, resp_ids.unsqueeze(-1)).squeeze(-1) # (resp_len,) # log-prob của token tiếp theo

    # eval per epoch
    def evaluate(self, val_dataset, batch_size: int = 16):
        self.model.eval()
        correct = 0
        total = len(val_dataset)
        batch_size = self.cfg.eval_batch_size if hasattr(self.cfg, "eval_batch_size") else batch_size
        log.info(f"Đang evaluation trên {total} samples với batch_size = {batch_size}...")
        scorer = self.rm_scorer if isinstance(self.rm_scorer, RuleBasedRewardScorer) else RuleBasedRewardScorer(self.device)
        
        for i in tqdm(range(0, total, batch_size), desc="Evaluating"):
            batch_examples = val_dataset[i : i + batch_size]
            
            batch_prompts = [example["prompt"] for example in batch_examples]
            batch_gold_answers = [example.get("gold_answer", "") for example in batch_examples]
            
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_prompt_len
            ).to(self.device)
            
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=self.cfg.bf16):
                    gen_out = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.cfg.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            
            for j, example in enumerate(batch_examples):
                prompt_len = input_ids.shape[1]
                resp_ids = gen_out[j, prompt_len:]
                
                pad_mask = (resp_ids != self.tokenizer.eos_token_id) & (resp_ids != self.tokenizer.pad_token_id)
                actual_len = max(pad_mask.sum().item(), 1)
                
                resp_text = self.tokenizer.decode(resp_ids[:actual_len], skip_special_tokens=True)
                
                # score
                pred_ans = scorer.extract_boxed_answer(resp_text)
                if scorer.normalize_answer(pred_ans) == scorer.normalize_answer(batch_gold_answers[j]) and batch_gold_answers[j]:
                    correct += 1
            
            # Giải phóng bộ nhớ đệm GPU sau mỗi batch
            del inputs, gen_out
            torch.cuda.empty_cache()
            
        acc = (correct / total) * 100
        log.info(f"Evaluated. Accuracy = {acc:.2f}% ({correct}/{total})")
        
        self.model.train()
        return acc

    # training loop
    def train(self, prompt_batch_size: int = 16):
        prompt_batch_size = self.cfg.train_batch_size if hasattr(self.cfg, "train_batch_size") else prompt_batch_size
        dataset_builder = MathDatasetBuilder(self.cfg.dataset_name, self.cfg.max_prompt_len)
        
        train_dataset = dataset_builder.load_train_data(self.tokenizer)
        val_dataset = dataset_builder.load_val_data(self.tokenizer)

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        
        total_batches = math.ceil(len(train_dataset) / prompt_batch_size)
        total_steps = self.cfg.num_epochs * total_batches
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(self.cfg.warmup_ratio * total_steps), total_steps)
        
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        global_step, acc_loss, acc_reward = 0, 0.0, 0.0

        for epoch in range(self.cfg.num_epochs):
            random.shuffle(train_dataset)
            log.info(f"Epoch {epoch + 1}/{self.cfg.num_epochs} - Tổng số batches: {total_batches}, Batch size: {prompt_batch_size}")
            
            pbar = tqdm(range(0, len(train_dataset), prompt_batch_size), desc=f"Epoch {epoch + 1}/{self.cfg.num_epochs}")

            for step_idx in pbar:
                batch_examples = train_dataset[step_idx : step_idx + prompt_batch_size]
                B = len(batch_examples)
                
                prompts = [ex["prompt"] for ex in batch_examples]
                gold_answers = [ex.get("gold_answer", "") for ex in batch_examples]

                inputs = self.tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=self.cfg.max_prompt_len
                ).to(self.device)
                
                # [B, seq_len] -> [BxG, seq_len]
                input_ids = inputs["input_ids"].repeat_interleave(self.cfg.G, dim=0)
                attention_mask = inputs["attention_mask"].repeat_interleave(self.cfg.G, dim=0)
                prompt_len = input_ids.shape[1]

                self.model.eval()
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=self.cfg.bf16):
                        gen_out = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=self.cfg.max_new_tokens,
                            temperature=self.cfg.temperature,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                self.model.train()

                lengths, masks, full_seqs, resp_texts = [], [], [], []
                total_generated = B * self.cfg.G
                
                for i in range(total_generated):
                    resp_ids = gen_out[i, prompt_len:]
                    pad_mask = (resp_ids != self.tokenizer.eos_token_id) & (resp_ids != self.tokenizer.pad_token_id)
                    actual_len = max(pad_mask.sum().item(), 1)
                    
                    lengths.append(actual_len)
                    masks.append(pad_mask.to(self.device))
                    full_seqs.append(gen_out[i])
                    resp_texts.append(self.tokenizer.decode(resp_ids[:actual_len], skip_special_tokens=True))

                # reward & advantage per group
                all_advs = []
                batch_mean_reward = 0.0
                
                for b in range(B):
                    # G responses per prompt
                    start_idx = b * self.cfg.G
                    end_idx = start_idx + self.cfg.G
                    group_texts = resp_texts[start_idx : end_idx]
                    gold = gold_answers[b]
                    
                    if isinstance(self.rm_scorer, RuleBasedRewardScorer):
                        rewards = self.rm_scorer.get_scores(group_texts, gold)
                    else:
                        rewards = self.rm_scorer.get_reward([prompts[b]] * self.cfg.G, group_texts)
                        
                    if not isinstance(rewards, torch.Tensor):
                        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                        
                    batch_mean_reward += rewards.mean().item()

                    if torch.all(rewards == rewards[0]):
                        all_advs.extend([0.0] * self.cfg.G)
                    else:
                        group_lens = lengths[start_idx : end_idx]
                        advs = compute_lpro_advantages(rewards.tolist(), group_lens, self.cfg.lambda_len)
                        all_advs.extend(advs.tolist())
                        
                batch_mean_reward /= B

                # loss + backprop
                total_loss_sum = torch.tensor(0.0, device=self.device)
                total_n_tokens = 0

                for i in range(total_generated):
                    if all_advs[i] == 0.0:
                        continue
                        
                    seq = full_seqs[i].unsqueeze(0).to(self.device)[:, :prompt_len + lengths[i]] # cut padding
                    mask_i = masks[i][:lengths[i]]

                    with torch.amp.autocast('cuda', enabled=self.cfg.bf16):
                        new_lp = self.get_resp_log_probs(self.model, seq, prompt_len, no_grad=False)
                        old_lp = self.get_resp_log_probs(self.ref_model, seq, prompt_len, no_grad=True)

                    loss_sum, n_valid = compute_dapo_token_loss(
                        new_lp, old_lp.detach(), float(all_advs[i]), mask_i, self.cfg.eps_low, self.cfg.eps_high
                    )
                    total_loss_sum += loss_sum
                    total_n_tokens += n_valid

                if total_n_tokens == 0: 
                    del gen_out, inputs, input_ids
                    torch.cuda.empty_cache()
                    continue
                
                # loss averaged over all tokens in the batch
                loss = total_loss_sum / total_n_tokens
                loss.backward()

                acc_loss += loss.item()
                acc_reward += batch_mean_reward

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "reward": f"{batch_mean_reward:.2f}"
                })

                # optimizer
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging & Saving
                if global_step % self.cfg.log_steps == 0:
                    log.info(f"Step {global_step} | Loss: {acc_loss/self.cfg.log_steps:.4f} | RM Score: {acc_reward/self.cfg.log_steps:.3f}")
                    acc_loss, acc_reward = 0.0, 0.0

                if global_step % self.cfg.save_steps == 0:
                    self.model.save_pretrained(os.path.join(self.cfg.output_dir, f"ckpt-{global_step}"))
                    self.tokenizer.save_pretrained(os.path.join(self.cfg.output_dir, f"ckpt-{global_step}"))

                del gen_out, inputs, input_ids, seq, new_lp, old_lp, loss_sum, total_loss_sum
                torch.cuda.empty_cache()
            
            log.info(f"Hoàn thành Train Epoch {epoch + 1}. Bắt đầu Eval...")
            self.evaluate(val_dataset)

        log.info("Training hoàn tất. Đang lưu model...")
        
        # save
        self.model.save_pretrained(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)

        # push to hub
        if getattr(self.cfg, "push_to_hub", False):
            if not self.cfg.hf_token:
                log.error("Chưa cung cấp hf_token trong Config.")
            else:
                log.info(f"Đang đẩy model lên Hugging Face Hub ({self.cfg.hub_repo_id})...")
                try:
                    self.model.push_to_hub(
                        self.cfg.hub_repo_id, 
                        token=self.cfg.hf_token,
                        commit_message="Upload Nexus Qwen-Math weights"
                    )
                    self.tokenizer.push_to_hub(
                        self.cfg.hub_repo_id, 
                        token=self.cfg.hf_token,
                        commit_message="Upload tokenizer"
                    )
                    log.info("Đã đẩy model lên Hugging Face Hub thành công!")
                except Exception as e:
                    log.error(f"Có lỗi xảy ra khi đẩy lên Hub: {e}")