# app.py
import streamlit as st
import requests
import json
from datetime import datetime
import re
import html

# Page configuration
st.set_page_config(
    page_title="Math Solver AI",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS với hỗ trợ LaTeX
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        max-height: 500px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 12px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        padding: 12px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #4CAF50;
    }
    .status-success {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-error {
        color: #F44336;
        font-weight: bold;
    }
    .stForm {
        border: 0 !important;
    }
    /* Cải thiện hiển thị LaTeX */
    .katex { 
        font-size: 1.1em !important;
    }
    .step-container {
        margin: 15px 0;
        padding: 10px;
        border-left: 3px solid #4CAF50;
        background-color: #f8f9fa;
    }
    /* Hiển thị boxed answer đẹp hơn */
    .latex-boxed {
        display: block;
        margin: 20px 0;
        padding: 15px;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-radius: 10px;
        border: 3px solid #FFC107;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Hàm chuyển đổi LaTeX và xử lý HTML
def format_latex_response(text):
    """
    Định dạng response từ model để hiển thị LaTeX đúng cách trong Streamlit.
    Pipeline xử lý:
      1. Unescape HTML entities
      2. Loại bỏ HTML tags
      3. Chuẩn hoá delimiter LaTeX (\\[...\\] → $$...$$, \\(...\\) → $...$)
      4. Sửa các lỗi LaTeX phổ biến từ model output
      5. Xử lý \\boxed{} cho hiển thị đẹp
      6. Định dạng các bước giải
    """
    if not text:
        return ""

    # 1. Xử lý HTML entities
    text = html.unescape(text)

    # 2. Loại bỏ tất cả các thẻ HTML
    text = re.sub(r'<[^>]*>', '', text)

    # 3. Chuẩn hoá delimiter LaTeX
    # \[ ... \] → $$ ... $$
    text = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    # \( ... \) → $ ... $
    text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)

    # 4. Sửa các lỗi LaTeX phổ biến từ model
    # 4a. "boxed{" không có backslash → "\boxed{"
    text = re.sub(r'(?<!\\)\bboxed\{', r'\\boxed{', text)
    # 4b. "$boxed{" hoặc "$$boxed{" → bỏ $ thừa, thêm \boxed
    text = re.sub(r'\${1,3}boxed\{', r'\\boxed{', text)
    # 4c. Loại bỏ trailing reference như $[3] sau boxed
    text = re.sub(r'(\\boxed\{[^}]*\})\$?\[\d+\]', r'\1', text)

    # 5. Tách \boxed{...} ra khỏi $ hoặc $$ wrapper trước khi xử lý
    # $\boxed{...}$ → \boxed{...}   (bỏ inline math wrapper)
    # $$\boxed{...}$$ → \boxed{...} (bỏ display math wrapper)
    boxed_inner = r'\\boxed\{(?:[^{}]|\{[^{}]*\})*\}'
    text = re.sub(r'\${1,2}(' + boxed_inner + r')\${1,2}[.,;:!?\s]*', r'\1', text)

    # 6. Xử lý \boxed{...} — thay bằng marker để render riêng bằng st.latex()
    BOXED_MARKER = '%%%BOXED%%%'
    def process_boxed(match):
        content = match.group(1).strip()
        # Loại bỏ $ thừa bên trong
        content = re.sub(r'^\$+|\$+$', '', content).strip()
        # Loại bỏ [number] references
        content = re.sub(r'\[\d+\]', '', content).strip()
        if not content:
            return ''
        return f'\n\n{BOXED_MARKER}{content}{BOXED_MARKER}\n\n'

    # Pattern hỗ trợ nested braces 1 cấp — cũng ăn trailing punctuation
    boxed_pattern = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}[.,;:!?]*'
    text = re.sub(boxed_pattern, process_boxed, text, flags=re.DOTALL)

    # 6. Sửa boxed không hoàn chỉnh (thiếu dấu })
    text = re.sub(r'\\boxed\{([^}]*)$', r'\\boxed{\1}', text, flags=re.MULTILINE)

    # 7. Đảm bảo display math ($$) có dòng trống bao quanh
    text = re.sub(r'(?<!\n)\n(\$\$)', r'\n\n\1', text)
    text = re.sub(r'(\$\$)\n(?!\n)', r'\1\n\n', text)

    # 8. Định dạng các bước giải (1., 2., ...) thành bold
    text = re.sub(r'^(\d+\..*)$', r'\n**\1**', text, flags=re.MULTILINE)
    # Thêm khoảng cách giữa các bước
    text = re.sub(r'(\n\*\*\d+\.)', r'\n\1', text)

    # 9. Xóa dòng chỉ chứa dấu chấm câu bị tách ra (orphan punctuation)
    text = re.sub(r'^\s*[.,;:!?]\s*$', '', text, flags=re.MULTILINE)

    # 10. Xử lý dòng trống thừa
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def render_response(formatted_text):
    """
    Render formatted text, xử lý boxed markers bằng st.latex() để căn giữa.
    Gọi hàm này thay vì st.markdown() trực tiếp.
    """
    BOXED_MARKER = '%%%BOXED%%%'
    if BOXED_MARKER not in formatted_text:
        st.markdown(formatted_text)
        return
    
    parts = formatted_text.split(BOXED_MARKER)
    # parts sẽ là: [text_before, boxed_content, text_after, boxed_content2, text_after2, ...]
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if i % 2 == 1:
            # Đây là nội dung boxed → render bằng st.latex() (tự động căn giữa)
            st.latex(r'\boxed{' + part + '}')
        else:
            # Text thường → render bằng markdown
            st.markdown(part)

def extract_boxed_content(boxed_string):
    """Trích xuất nội dung từ chuỗi boxed bị lỗi"""
    # Tìm nội dung giữa { và }
    match = re.search(r'\{([^{}]*)\}', boxed_string)
    if match:
        content = match.group(1)
        # Loại bỏ các ký tự không mong muốn
        content = re.sub(r'\$+', '', content)
        content = re.sub(r'\[.*?\]', '', content)
        return content.strip()
    return ""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = ""
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""

# Sidebar
with st.sidebar:
    st.title("⚙️ Cấu hình")
    
    # API Configuration
    st.subheader("API Configuration")
    api_url = st.text_input(
        "Colab API URL",
        value=st.session_state.api_url,
        placeholder="https://xxxx-xxxx-xxxx.ngrok-free.app",
        help="Nhập URL từ Google Colab (ngrok)"
    )
    
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url.rstrip('/')
        st.success(f"Connected to: Dat1710/Nexus-1.5B")
    
    st.divider()
    
    # Model Parameters
    st.subheader("Model Parameters")
    
    reasoning_method = st.selectbox(
        "Reasoning Method",
        ["CoT", "TIR"],
        index=0,
        help="CoT: Chain-of-Thought\nTIR: Tool-Integrated Reasoning"
    )
    
    max_tokens = st.slider(
        "Max New Tokens",
        min_value=128,
        max_value=2048,
        value=1024,
        step=128
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1
    )
    
    top_p = st.slider(
        "Top-p (Nucleus sampling)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05
    )
    
    custom_system = st.text_area(
        "Custom System Message (Optional)",
        height=100,
        help="Ghi đè system message mặc định"
    )
    
    st.divider()
    
    # Examples với LaTeX
    st.subheader("📚 Ví dụ")
    examples = [
        "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$.",
        "Solve the quadratic equation: $x^2 - 5x + 6 = 0$",
        "What is the derivative of $f(x) = 3x^4 + 2x^2 - 5x + 7$?",
        "Calculate the integral: $\\int (3x^2 + 2x - 1) dx$",
        "Find the limit: $\\lim_{x \\to 0} \\frac{\\sin(x)}{x}$"
    ]
    
    for example in examples:
        if st.button(f"📝 {example[:50]}..." if len(example) > 50 else f"📝 {example}"):
            st.session_state.current_prompt = example
            st.rerun()
    
    st.divider()
    
    # Hiển thị thông tin LaTeX
    with st.expander("ℹ️ Hướng dẫn LaTeX"):
        st.markdown("""
        **Hỗ trợ LaTeX:**
        - `$...$`: Công thức inline (ví dụ: `$x^2$`)
        - `$$...$$`: Công thức display (ví dụ: `$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$`)
        - `\\boxed{...}`: Hiển thị đáp án trong hộp
        
        **Ví dụ:**
        - Phương trình: `$x^2 + y^2 = r^2$`
        - Tích phân: `$$\\int_a^b f(x) dx$$`
        - Đáp án: `\\boxed{x = -1}`
        """)
    
    # Clear chat button
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main content
st.markdown('<h1 class="main-header">🧮 Math Solver AI Assistant</h1>', unsafe_allow_html=True)

# Connection status
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.session_state.api_url:
        try:
            # Kiểm tra health endpoint
            response = requests.get(f"{st.session_state.api_url}/health", timeout=15)
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ Connected to Dat1710/Nexus-1.5B")
            else:
                st.error("❌ Connection failed")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to server")
        except Exception as e:
            st.warning(f"⚠️ Connection error: {str(e)}")
    else:
        st.info("ℹ️ Please enter API URL from Colab in sidebar")

# Chat container
st.markdown("### 💬 Chat")
chat_container = st.container()

# Display chat messages với hỗ trợ LaTeX
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                content = message["content"]
                
                # Format LaTeX và hiển thị
                formatted_content = format_latex_response(content)
                render_response(formatted_content)
                
            if "response_data" in message:
                with st.expander("📊 Response Details"):
                    st.json(message["response_data"])

# Input area
with st.form(key="input_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        prompt = st.text_area(
            "Nhập bài toán:",
            value=st.session_state.current_prompt,
            placeholder="Nhập bài toán toán học của bạn ở đây (có thể dùng LaTeX như $x^2 + y^2 = 1$)...",
            height=100,
            key="prompt_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("🚀 Gửi", use_container_width=True, type="primary")

# Process input
if submit_button and prompt.strip():
    # Reset current prompt
    st.session_state.current_prompt = ""
    
    # Add user message
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    
    # Hiển thị câu hỏi dạng LaTeX preview trước khi gửi request
    with st.chat_message("user"):
        st.markdown("**LaTeX preview:**")
        st.markdown(format_latex_response(prompt))
    
    # Check connection
    if not st.session_state.api_url:
        st.error("Vui lòng nhập API URL trong sidebar trước!")
        st.rerun()
    
    # Show assistant placeholder
    with st.chat_message("assistant"):
        status_msg = st.empty()
        status_msg.markdown("⏳ Đang xử lý...")
        response_container = st.container()
        
        # Prepare request
        request_data = {
            "prompt": prompt,
            "reasoning_method": reasoning_method,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        if custom_system.strip():
            request_data["system_message"] = custom_system
        
        try:
            # Send request to Colab API
            response = requests.post(
                f"{st.session_state.api_url}/generate",
                json=request_data,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "")
                
                # Format response với LaTeX
                formatted_response = format_latex_response(raw_response)
                
                # Xóa status và hiển thị response
                status_msg.empty()
                with response_container:
                    render_response(formatted_response)
                
                # Add assistant message with metadata
                assistant_message = {
                    "role": "assistant",
                    "content": raw_response,
                    "formatted_content": formatted_response,
                    "response_data": {
                        "model": result.get("model"),
                        "status": result.get("status"),
                        "timestamp": datetime.now().isoformat(),
                        "parameters": result.get("parameters", {})
                    }
                }
                st.session_state.messages.append(assistant_message)
                
            else:
                error_msg = f"❌ Lỗi API: {response.status_code} - {response.text}"
                status_msg.markdown(error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "⏰ Request timeout. Model might be taking too long."
            message_placeholder.markdown(error_msg)
            
        except Exception as e:
            error_msg = f"❌ Lỗi kết nối: {str(e)}"
            message_placeholder.markdown(error_msg)
    
    # Force rerun để cập nhật giao diện
    st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Powered by Qwen2.5-Math-1.5B-Instruct • Running on Google Colab GPU • Streamlit Frontend</p>
    <p style='font-size: 0.9em;'>📚 Hỗ trợ LaTeX đầy đủ: $...$ cho inline, $$...$$ cho display, \\boxed{} cho đáp án</p>
</div>
""", unsafe_allow_html=True)