import streamlit as st
from text_generation import memory_efficient_generate
import time

def main():
    if not st.session_state.training_complete or st.session_state.model is None:
        st.info("Complete training on the Training page to unlock text generation!")
        return
    
    st.header("Text Generation")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            value="<|begin_of_text|><|start_header_id|>start<|end_header_id|>\nWhat is artificial intelligence used for?\n<|start_header_id|>end<|end_header_id|>\n<|start_header_id|>reply<|end_header_id|>\n",
            height=100
        )
    
    with col2:
        max_gen_length = st.slider("Max length", 20, 100, 50)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
    
    if st.button("Generate Text"):
        with st.spinner("Generating..."):
            try:
                gen_start = time.time()
                generated_text = memory_efficient_generate(
                    st.session_state.model,
                    st.session_state.tokenizer,
                    prompt,
                    max_gen_length,
                    temperature
                )
                gen_time = time.time() - gen_start
                st.subheader("Generated Text:")
                st.code(generated_text, language="text")
                st.caption(f"Generation time: {gen_time:.2f}s")
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

if __name__ == "__main__":
    main()