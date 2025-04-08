import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

# Load data
if "df" not in st.session_state:
    df = pd.read_csv("temp_data/bs_gpt_resp.csv", dtype={"gpt_subtopic": str})
    if "user_subtopic" not in df.columns:
        df["user_subtopic"] = ""
    st.session_state.df = df

# Accuracy and Save bar - visible on all views
with st.container():
    bar = st.columns([6, 1])
    with bar[0]:
        valid_rows = st.session_state.df["user_subtopic"].astype(str).str.strip() != ""
        num_filled = valid_rows.sum()
        if num_filled > 0:
            correct = (st.session_state.df.loc[valid_rows, "gpt_subtopic"].astype(str).str.strip() ==
                       st.session_state.df.loc[valid_rows, "user_subtopic"].astype(str).str.strip()).sum()
            accuracy = correct / num_filled * 100
            st.markdown(f"✅ **Accuracy: {accuracy:.1f}%** ({correct}/{num_filled})")
        else:
            st.markdown("🟡 No user subtopics entered yet.")
    with bar[1]:
        if st.button("💾 Save to CSV"):
            st.session_state.df.to_csv("temp_data/bs_gpt_resp_labeled.csv", index=False)
            st.success("Saved labeled data to `bs_gpt_resp_labeled.csv`")

view = st.radio("Choose View", ["Single Post View", "Review All"])
df = st.session_state.df

# Sidebar codebook for single post view
if view == "Single Post View":
    with st.sidebar:
        st.markdown("## Codebook")
        st.markdown("""
        1. Crime  
            1.1. Crime generally  

        2. Environment  
            2.1. Climate change  
            2.2. Other environmental issues  

        3. Immigration  
            3.1. Immigration generally, including anything about ICE or deportations  

        4. Social issues  
            4.1. Abortion and reproductive health  
            4.2. Guns and gun control  
            4.3. LGBTQ+ issues, including transgender issues  
            4.4. Racial issues, including affirmative action and racial discrimination, as well as discussion of \"DEI,\" diversity or \"illegal discrimination\"  
            4.5. Education  
            4.6. Free speech, the news media, and government restrictions or threats of restrictions  
            4.7. Other social issues, including culture war issues, relationships between the genders (but not trans-gender), religion and society, labor, and other social issues that are not covered above  

        5. Public health  
            5.1. Covid, including covid vaccines  
            5.2. Other vaccines  
            5.3 MAHA, or Make America Healthy Again  
            5.4 Cutting or canceling of federal medical research grants (but not staff)  
            5.5. Other public health issues  

        6. Economy  
            6.1. Economy generally - including inflation, housing, jobs, cost of living, and other macro-economic issues.  

        7. Technology  
            7.1. AI, LLMs  
            7.2. Crypto  
            7.3. Other technology issues  

        8. Government, politics and elections  
            8.1. DOGE, or the Department of Government Efficiency  
            8.2. Nominations and senate approval of cabinet members  
            8.3. Federal budget approval in Congress and government potentially shutting down  
            8.4. National security leaks  
            8.5. Other political or government related posts that do not fit into other categories  

        9. International issues  
            9.1. Israel, Gaza or Palestine  
            9.2. Ukraine war  
            9.3. Anything outside the United States or involve US foreign relations except for Israel, Gaza, Ukraine, or immigration.  

        10. No topic  
            10.1. None of the above topics, or non-political  
        """)

    st.title("🧠 GPT Subtopic Reviewer - One by One")

    if "row_index" not in st.session_state:
        st.session_state.row_index = 0

    current_index = st.session_state.row_index
    row = df.iloc[current_index]

    post_col, input_col = st.columns([3, 2], gap="large")

    with input_col:
        key = f"user_input_{current_index}"
        user_value = df.at[current_index, "user_subtopic"]
        new_value = st.text_input("✏️ Your Subtopic", value=user_value, key=key)
        st.session_state.df.at[current_index, "user_subtopic"] = new_value

        if new_value.strip():
            st.markdown(f"<div style='margin-top:0.5em; font-size:0.9em; color:gray;'>🤖 <strong>GPT Prediction:</strong> {row['gpt_subtopic']}</div>", unsafe_allow_html=True)

        st.markdown("---")
        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            if st.button("⬅️ Previous"):
                st.session_state.row_index = max(0, current_index - 1)
                st.rerun()

        with col_next:
            if st.button("💾 Save and Next"):
                st.session_state.row_index = (current_index + 1) % len(df)
                st.rerun()

        if st.button("📋 Show All"):
            st.session_state.view_all = True
            st.rerun()

    with post_col:
        st.components.v1.html(row["html"], scrolling=True, height=1600)

    st.markdown(f"Showing {current_index + 1} of {len(df)}")

elif view == "Review All":
    st.title("📋 Review All Entries")

    def get_row_style(row):
        user = str(row["user_subtopic"]).strip()
        gpt = str(row["gpt_subtopic"]).strip()
        if user and user != gpt:
            return "background-color: #fff3cd;"
        return ""

    styled_df = df[["full_text", "gpt_subtopic", "user_subtopic"]].copy()
    styled = styled_df.style.apply(lambda row: [get_row_style(row)] * len(row), axis=1)

    st.dataframe(styled, use_container_width=True, height=800)

    edited_df = st.data_editor(
        df[["full_text", "gpt_subtopic", "user_subtopic"]],
        column_config={
            "full_text": st.column_config.TextColumn(label="Post", help="The original post text with line breaks", disabled=True),
            "gpt_subtopic": st.column_config.TextColumn(label="🤖 GPT Subtopic", help="The subtopic predicted by GPT"),
            "user_subtopic": st.column_config.TextColumn(label="✏️ Your Subtopic", help="You can correct GPT's label here"),
        },
        use_container_width=True,
        hide_index=True,
        height=800
    )

    st.session_state.df["user_subtopic"] = edited_df["user_subtopic"]
    st.session_state.df.to_csv("temp_data/bs_gpt_resp_labeled.csv", index=False)
