# app.py

# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import numpy as np
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st

# -----------------------------
# 1️⃣ Load FAQ dataset
# -----------------------------
faq_data = [
    {"question": "What is the full name of WIT?", "answer": "WIT stands for Women's Institute of Technology, Darbhanga located in Bihar."},
    {"question": "What is the campus address?", "answer": "WIT Darbhanga, Opp. Raj Dental College, NH-27, Darbhanga, Bihar."},
    {"question": "What is the official website of WIT?", "answer": "The official website of WIT Darbhanga is https://www.wit.ac.in/."},
    {"question": "How can I contact the WIT campus?", "answer": "You can reach WIT Darbhanga via phone at the main campus helpdesk or through their official website contact page."},
    {"question": "What undergraduate programs are offered?", "answer": "WIT offers B.Tech programs in CSE, IT, ECE, and other engineering branches as per the academic calendar."},
    {"question": "What is the eligibility for B.Tech admission?", "answer": "Students must have passed 12th with Physics, Chemistry, and Math with minimum qualifying marks as per university norms."},
    {"question": "Does WIT accept JEE score?", "answer": "Yes, B.Tech admissions may consider JEE Main score based on the prevailing admission guidelines for the academic year."},
    {"question": "How do I apply for admission?", "answer": "Visit the WIT website admissions section and fill the online application form or contact the admissions office."},
    {"question": "What documents are required for admission?", "answer": "Common documents include 10th & 12th marksheets, transfer certificate, migration certificate, Aadhaar, photos, and entrance exam scorecards."},
    {"question": "What are the hostel facilities available?", "answer": "WIT provides safe on-campus hostel facilities with mess, security, and essential amenities for students."},
    {"question": "How can I apply for hostel accommodation?", "answer": "Students can apply through the hostel form available at the Student Services Office or email the hostel administration."},
    {"question": "Are hostel rooms furnished?", "answer": "Yes, rooms are furnished with bed, table, chair, and basic utilities."},
    {"question": "What are the hostel rules?", "answer": "Hostel rules include attendance, curfew timings, visitor restrictions, and maintaining discipline inside the campus."},
    {"question": "What are the library hours?", "answer": "The library is generally open from 8:30 AM to 6:00 PM on weekdays and Saturdays."},
    {"question": "Does the campus have WiFi?", "answer": "Yes, the campus provides high-speed WiFi for students and faculty."},
    {"question": "How can students get WiFi access?", "answer": "Students must register their devices at the ICT department to receive login credentials."},
    {"question": "What labs are available in WIT?", "answer": "WIT has well-equipped labs for Computer Science, Electronics, Applied Sciences, and Engineering departments."},
    {"question": "How can I contact the placement cell?", "answer": "Students can email the placement office or visit the Training & Placement cell in the administration block."},
    {"question": "Does the college provide placement support?", "answer": "Yes, the placement cell provides training, internships, industry sessions, and placement opportunities."},
    {"question": "Which companies visit WIT for placements?", "answer": "Companies from IT, software, electronics, and consulting domains frequently visit for campus recruitment."},
    {"question": "Is there a canteen in WIT?", "answer": "Yes, WIT has a clean and well-maintained canteen offering snacks, meals, and beverages."},
    {"question": "What sports facilities are available?", "answer": "The campus provides indoor and outdoor sports facilities including badminton, volleyball, and athletics."},
    {"question": "Does WIT organize cultural programs?", "answer": "Yes, WIT conducts annual cultural events, tech fests, seminars, and workshops."},
    {"question": "How to get a bonafide certificate?", "answer": "Students can request a bonafide certificate through the administrative office by filling a simple form."},
    {"question": "How to apply for leave?", "answer": "Students must submit a handwritten or online leave application to the department coordinator."},
    {"question": "What is the dress code in WIT?", "answer": "Students are expected to follow formal or semi-formal attire guidelines as instructed by the college."},
    {"question": "Is attendance compulsory?", "answer": "Yes, the minimum attendance requirement is usually 75% as per university regulations."},
    {"question": "Where can I see the academic calendar?", "answer": "The academic calendar is available on the official WIT website under the academics section."},
    {"question": "How to check exam schedules?", "answer": "Exam schedules are published on the notice board and on the official website prior to examinations."},
    {"question": "How to access study materials?", "answer": "Students can access study materials through the library, department Google Classroom, or faculty-provided resources."},
    {"question": "Does WIT have a grievance cell?", "answer": "Yes, WIT has an online and offline grievance redressal system for students."},
    {"question": "How to report technical issues on campus?", "answer": "Students can contact the ICT desk for laptop, WiFi, or system-related issues."},
    {"question": "Is transportation provided?", "answer": "Local transport options are available around campus; students may also use private transportation."},
    {"question": "How to request a transcript?", "answer": "Students can apply for transcripts through the examination cell by submitting a formal request."},
    {"question": "Does WIT provide internship support?", "answer": "Yes, the Training & Placement cell assists students in securing internships."},
    {"question": "How to update personal details?", "answer": "Students can update details by submitting an application at the administrative office."},
    {"question": "What is the fee payment procedure?", "answer": "Fees can be paid online through the WIT portal or offline at the accounts department."},
    {"question": "Where can I see fee deadlines?", "answer": "Fee deadlines are notified via email, notice boards, and the official website."},
    {"question": "Does the college have medical facilities?", "answer": "Yes, basic medical assistance is available on campus with first-aid support."},
    {"question": "How to contact faculty members?", "answer": "Students can reach faculty through institutional email or meet during official office hours."}
]

# Create DataFrame
df = pd.DataFrame(faq_data)
df['q_clean'] = df['question'].apply(
    lambda x: re.sub(r'\s+', ' ', x.lower()).translate(str.maketrans('', '', string.punctuation)).strip()
)

# -----------------------------
# 2️⃣ Load Sentence Transformer & generate embeddings
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
faq_embeddings = model.encode(df['q_clean'].tolist(), convert_to_numpy=True)

# -----------------------------
# 3️⃣ Semantic Search Function
# -----------------------------
def get_answer_semantic(user_query, top_k=1, min_sim=0.5):
    query_clean = re.sub(r'\s+', ' ', user_query.lower()).translate(str.maketrans('', '', string.punctuation)).strip()
    query_emb = model.encode([query_clean], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, faq_embeddings).flatten()
    idx = sims.argsort()[::-1][0]
    score = sims[idx]
    if score >= min_sim:
        return {'answer': df.loc[idx, 'answer'], 'score': float(score), 'matched_question': df.loc[idx, 'question']}
    else:
        top_idxs = sims.argsort()[::-1][:3]
        suggestions = df.loc[top_idxs, 'question'].tolist()
        return {'answer': None, 'score': float(score), 'suggestions': suggestions}

# -----------------------------
# 4️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="WIT Darbhanga Semantic Chatbot", page_icon="🤖")
st.title("🤖 WIT Darbhanga Semantic Chatbot")
st.markdown("Ask me anything about WIT Darbhanga: admissions, hostel, library, placements, WiFi, etc.")

user_input = st.text_input("You:")

if user_input:
    resp = get_answer_semantic(user_input)
    if resp['answer']:
        st.markdown(f"**Bot:** {resp['answer']}  _(score: {resp['score']:.2f})_")
        st.markdown(f"*Matched Question: {resp['matched_question']}*")
    else:
        st.markdown("**Bot:** Sorry, I couldn't find an exact answer. Did you mean:")
        for s in resp['suggestions']:
            st.markdown(f"- {s}")

# Optional: Show all FAQs
with st.expander("View all FAQs"):
    for i, row in df.iterrows():
        st.markdown(f"**Q:** {row['question']}  \n**A:** {row['answer']}")
