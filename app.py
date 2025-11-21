{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# WIT Darbhanga Semantic Chatbot\n",
    "This notebook implements a **semantic search FAQ chatbot** using **Sentence-Transformers** and **Streamlit** for a nice UI.\n",
    "- 40+ campus FAQs\n",
    "- Semantic embeddings for query matching\n",
    "- Streamlit interactive interface"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Imports\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import re\n",
    "import string\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from sentence_transformers import SentenceTransformer\n",
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# -----------------------------\n",
    "# 1️⃣ Load FAQ dataset\n",
    "# -----------------------------\n",
    "faq_data = [\n",
    "    {\"question\": \"What is the full name of WIT?\", \"answer\": \"WIT stands for Women's Institute of Technology, Darbhanga located in Bihar.\"},\n",
    "    {\"question\": \"What is the campus address?\", \"answer\": \"WIT Darbhanga, Opp. Raj Dental College, NH-27, Darbhanga, Bihar.\"},\n",
    "    {\"question\": \"What is the official website of WIT?\", \"answer\": \"The official website of WIT Darbhanga is https://www.wit.ac.in/.\"},\n",
    "    {\"question\": \"How can I contact the WIT campus?\", \"answer\": \"You can reach WIT Darbhanga via phone at the main campus helpdesk or through their official website contact page.\"},\n",
    "    {\"question\": \"What undergraduate programs are offered?\", \"answer\": \"WIT offers B.Tech programs in CSE, IT, ECE, and other engineering branches as per the academic calendar.\"},\n",
    "    {\"question\": \"What is the eligibility for B.Tech admission?\", \"answer\": \"Students must have passed 12th with Physics, Chemistry, and Math with minimum qualifying marks as per university norms.\"},\n",
    "    {\"question\": \"Does WIT accept JEE score?\", \"answer\": \"Yes, B.Tech admissions may consider JEE Main score based on the prevailing admission guidelines for the academic year.\"},\n",
    "    {\"question\": \"How do I apply for admission?\", \"answer\": \"Visit the WIT website admissions section and fill the online application form or contact the admissions office.\"},\n",
    "    {\"question\": \"What documents are required for admission?\", \"answer\": \"Common documents include 10th & 12th marksheets, transfer certificate, migration certificate, Aadhaar, photos, and entrance exam scorecards.\"},\n",
    "    {\"question\": \"What are the hostel facilities available?\", \"answer\": \"WIT provides safe on-campus hostel facilities with mess, security, and essential amenities for students.\"},\n",
    "    {\"question\": \"How can I apply for hostel accommodation?\", \"answer\": \"Students can apply through the hostel form available at the Student Services Office or email the hostel administration.\"},\n",
    "    {\"question\": \"Are hostel rooms furnished?\", \"answer\": \"Yes, rooms are furnished with bed, table, chair, and basic utilities.\"},\n",
    "    {\"question\": \"What are the hostel rules?\", \"answer\": \"Hostel rules include attendance, curfew timings, visitor restrictions, and maintaining discipline inside the campus.\"},\n",
    "    {\"question\": \"What are the library hours?\", \"answer\": \"The library is generally open from 8:30 AM to 6:00 PM on weekdays and Saturdays.\"},\n",
    "    {\"question\": \"Does the campus have WiFi?\", \"answer\": \"Yes, the campus provides high-speed WiFi for students and faculty.\"},\n",
    "    {\"question\": \"How can students get WiFi access?\", \"answer\": \"Students must register their devices at the ICT department to receive login credentials.\"},\n",
    "    {\"question\": \"What labs are available in WIT?\", \"answer\": \"WIT has well-equipped labs for Computer Science, Electronics, Applied Sciences, and Engineering departments.\"},\n",
    "    {\"question\": \"How can I contact the placement cell?\", \"answer\": \"Students can email the placement office or visit the Training & Placement cell in the administration block.\"},\n",
    "    {\"question\": \"Does the college provide placement support?\", \"answer\": \"Yes, the placement cell provides training, internships, industry sessions, and placement opportunities.\"},\n",
    "    {\"question\": \"Which companies visit WIT for placements?\", \"answer\": \"Companies from IT, software, electronics, and consulting domains frequently visit for campus recruitment.\"},\n",
    "    {\"question\": \"Is there a canteen in WIT?\", \"answer\": \"Yes, WIT has a clean and well-maintained canteen offering snacks, meals, and beverages.\"},\n",
    "    {\"question\": \"What sports facilities are available?\", \"answer\": \"The campus provides indoor and outdoor sports facilities including badminton, volleyball, and athletics.\"},\n",
    "    {\"question\": \"Does WIT organize cultural programs?\", \"answer\": \"Yes, WIT conducts annual cultural events, tech fests, seminars, and workshops.\"},\n",
    "    {\"question\": \"How to get a bonafide certificate?\", \"answer\": \"Students can request a bonafide certificate through the administrative office by filling a simple form.\"},\n",
    "    {\"question\": \"How to apply for leave?\", \"answer\": \"Students must submit a handwritten or online leave application to the department coordinator.\"},\n",
    "    {\"question\": \"What is the dress code in WIT?\", \"answer\": \"Students are expected to follow formal or semi-formal attire guidelines as instructed by the college.\"},\n",
    "    {\"question\": \"Is attendance compulsory?\", \"answer\": \"Yes, the minimum attendance requirement is usually 75% as per university regulations.\"},\n",
    "    {\"question\": \"Where can I see the academic calendar?\", \"answer\": \"The academic calendar is available on the official WIT website under the academics section.\"},\n",
    "    {\"question\": \"How to check exam schedules?\", \"answer\": \"Exam schedules are published on the notice board and on the official website prior to examinations.\"},\n",
    "    {\"question\": \"How to access study materials?\", \"answer\": \"Students can access study materials through the library, department Google Classroom, or faculty-provided resources.\"},\n",
    "    {\"question\": \"Does WIT have a grievance cell?\", \"answer\": \"Yes, WIT has an online and offline grievance redressal system for students.\"},\n",
    "    {\"question\": \"How to report technical issues on campus?\", \"answer\": \"Students can contact the ICT desk for laptop, WiFi, or system-related issues.\"},\n",
    "    {\"question\": \"Is transportation provided?\", \"answer\": \"Local transport options are available around campus; students may also use private transportation.\"},\n",
    "    {\"question\": \"How to request a transcript?\", \"answer\": \"Students can apply for transcripts through the examination cell by submitting a formal request.\"},\n",
    "    {\"question\": \"Does WIT provide internship support?\", \"answer\": \"Yes, the Training & Placement cell assists students in securing internships.\"},\n",
    "    {\"question\": \"How to update personal details?\", \"answer\": \"Students can update details by submitting an application at the administrative office.\"},\n",
    "    {\"question\": \"What is the fee payment procedure?\", \"answer\": \"Fees can be paid online through the WIT portal or offline at the accounts department.\"},\n",
    "    {\"question\": \"Where can I see fee deadlines?\", \"answer\": \"Fee deadlines are notified via email, notice boards, and the official website.\"},\n",
    "    {\"question\": \"Does the college have medical facilities?\", \"answer\": \"Yes, basic medical assistance is available on campus with first-aid support.\"},\n",
    "    {\"question\": \"How to contact faculty members?\", \"answer\": \"Students can reach faculty through institutional email or meet during official office hours.\"}\n",
    "]\n",
    "\n",
    "df = pd.DataFrame(faq_data)\n",
    "df['q_clean'] = df['question'].apply(lambda x: re.sub(r'\\s+', ' ', x.lower()).translate(str.maketrans('', '', string.punctuation)).strip())"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# -----------------------------\n",
    "# 2️⃣ Generate Sentence Embeddings\n",
    "# -----------------------------\n",
    "model = SentenceTransformer('all-MiniLM-L6-v2')\n",
    "faq_embeddings = model.encode(df['q_clean'].tolist(), convert_to_numpy=True)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# -----------------------------\n",
    "# 3️⃣ Semantic Search Function\n",
    "# -----------------------------\n",
    "def get_answer_semantic(user_query, top_k=1, min_sim=0.5):\n",
    "    query_clean = re.sub(r'\\s+', ' ', user_query.lower()).translate(str.maketrans('', '', string.punctuation)).strip()\n",
    "    query_emb = model.encode([query_clean], convert_to_numpy=True)\n",
    "    sims = cosine_similarity(query_emb, faq_embeddings).flatten()\n",
    "    idx = sims.argsort()[::-1][0]\n",
    "    score = sims[idx]\n",
    "    if score >= min_sim:\n",
    "        return {'answer': df.loc[idx, 'answer'], 'score': float(score), 'matched_question': df.loc[idx, 'question']}\n",
    "    else:\n",
    "        top_idxs = sims.argsort()[::-1][:3]\n",
    "        suggestions = df.loc[top_idxs, 'question'].tolist()\n",
    "        return {'answer': None, 'score': float(score), 'suggestions': suggestions}"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# -----------------------------\n",
    "# 4️⃣ Streamlit Chatbot UI\n",
    "# -----------------------------\n",
    "st.set_page_config(page_title=\"WIT Darbhanga Semantic Chatbot\", page_icon=\"🤖\")\n",
    "st.title(\"🤖 WIT Darbhanga Semantic Chatbot\")\n",
    "st.markdown(\"Ask me anything about WIT Darbhanga: admissions, hostel, library, placements, WiFi, etc.\")\n",
    "\n",
    "user_input = st.text_input(\"You:\")\n",
    "\n",
    "if user_input:\n",
    "    resp = get_answer_semantic(user_input)\n",
    "    if resp['answer']:\n",
    "        st.markdown(f\"**Bot:** {resp['answer']}  _(score: {resp['score']:.2f})_\")\n",
    "        st.markdown(f\"*Matched Question: {resp['matched_question']}*\")\n",
    "    else:\n",
    "        st.markdown(\"**Bot:** Sorry, I couldn't find an exact answer. Did you mean:\")\n",
    "        for s in resp['suggestions']:\n",
    "            st.markdown(f\"- {s}\")\n",
    "\n",
    "# Optional: Show all FAQs\n",
    "with st.expander(\"View all FAQs\"):\n",
    "    for i, row in df.iterrows():\n",
    "        st.markdown(f\"**Q:** {row['question']}  \\n**A:** {row['answer']}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
