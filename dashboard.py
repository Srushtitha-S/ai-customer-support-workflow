import sqlite3
import pandas as pd
import streamlit as st

from app import process_ticket, init_db, DB_PATH

st.set_page_config(
    page_title="AI Support Desk",
    page_icon="🎫",
    layout="wide",
)

init_db()

st.title("🎫 AI-Powered Customer Support Desk")
st.caption("Classify customer tickets, prioritize issues, and identify tickets needing human review.")

with st.form("ticket_form"):
    message = st.text_area(
        "Customer message",
        placeholder="Example: I was charged twice for my order.",
        height=120,
    )
    submitted = st.form_submit_button("Analyze ticket", type="primary")

if submitted:
    if not message.strip():
        st.warning("Please enter a customer message.")
    else:
        with st.spinner("AI is analyzing the ticket..."):
            try:
                result, status = process_ticket(message.strip())
                st.success("Ticket processed successfully")

                col1, col2, col3 = st.columns(3)
                col1.metric("Category", result.category.value)
                col2.metric("Priority", result.priority.value.upper())
                col3.metric(
                    "Human review",
                    "Required" if result.needs_human_review else "Not required",
                )

                st.write("**Summary:**", result.summary)
                st.write("**Assigned team:**", result.suggested_team)
            except Exception as exc:
                st.error(f"Could not process ticket: {exc}")

st.divider()
st.subheader("Ticket history")

connection = sqlite3.connect(DB_PATH)

try:
    tickets = pd.read_sql_query(
        """
        SELECT id, message, category, priority, summary,
               suggested_team, needs_human_review, status, created_at
        FROM tickets
        ORDER BY id DESC
        """,
        connection,
    )
finally:
    connection.close()

if tickets.empty:
    st.info("No tickets have been saved yet.")
else:
    total, high, review = st.columns(3)
    total.metric("Total tickets", len(tickets))
    high.metric(
        "High priority",
        int((tickets["priority"] == "high").sum()),
    )
    review.metric(
        "Human review required",
        int((tickets["needs_human_review"] == 1).sum()),
    )

    search = st.text_input("Search ticket messages")
    if search.strip():
        tickets = tickets[
            tickets["message"].str.contains(search, case=False, na=False)
        ]

    st.dataframe(tickets, use_container_width=True, hide_index=True)
