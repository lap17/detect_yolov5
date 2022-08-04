import streamlit as st
import sys
import gc
from detect_clothes import func_detect_clothes
from streamlit.legacy_caching import clear_cache

def main():
    gc.enable()
    st.header("Fashion Items Detection Demo")
    st.sidebar.markdown("""<center data-parsed=""><img src="http://drive.google.com/uc?export=view&id=1D-pN81CupHMcxb7xa5-Z6JZZIbagRqH_" align="center"></center>""",unsafe_allow_html=True,)
    st.sidebar.markdown(" ")
    
    def reload():
        clear_cache()
        gc.collect()
        st.experimental_rerun()

    pages = st.sidebar.columns([1, 1, 1])
    
    pages[0].markdown(" ")

    if pages[1].button("Reload App"):
        reload()
    func_detect_clothes()
    
main()