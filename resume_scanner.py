from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Define the directory where PDFs are stored
RESUME_DIR = r"C:\ResumeScanner(proj)"  # Use raw string to handle parentheses

# Function to clean text (remove special characters, extra whitespace)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s.,-@|]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return clean_text(text)

# Function to extract key details from resumes (Name, Email, Mobile)
def extract_resume_details(text):
    details = {"Name": "", "Email": "", "Mobile": ""}
    
    # Extract Name (search entire text for name-like pattern, more robust regex)
    lines = text.split("\n")
    for line in lines:
        # Matches names like "John Doe", "JOHN DOE", "Maria de la Cruz", "Mary-Jane Smith"
        name_match = re.search(r"\b([A-Za-z]+(?:[\s-][A-Za-z]+){1,3})\b", line.strip())
        if name_match:
            details["Name"] = name_match.group(1).strip()
            break
    
    if not details["Name"]:
        details["Name"] = "Unknown"
    
    # Extract Email
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    if email_match:
        details["Email"] = email_match.group(0)
    else:
        details["Email"] = "Not Found"
    
    # Extract Mobile Number
    mobile_match = re.search(r"(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}", text)
    if mobile_match:
        details["Mobile"] = mobile_match.group(0).replace(" ", "").replace("-", "").replace(".", "")
    else:
        details["Mobile"] = "Not Found"
    
    return details

# Function to extract skills
def extract_skills(text):
    skills_pattern = re.compile(r"\b(Python|Machine Learning|NLP|AI|Deep Learning|Data Science|SQL|Big Data|Hadoop|Spark|Power BI|Tableau|Data Engineering|AWS|GCP|Azure|Java|R|TensorFlow|PyTorch|Scikit-learn|Keras|Hugging Face|Excel|SAS|Matplotlib|Seaborn|D3.js|MySQL|PostgreSQL|MongoDB|Cassandra|Docker|Kubernetes|Jenkins|Git|Ansible|GenAI|LLMs|Snowflake|HTML5|CSS3|JavaScript|React.js|Node.js|Express.js|Cloudinary|MapBox|Multer|Bootstrap|EJS|RESTful|MERN|MVC)\b", re.IGNORECASE)
    return list(set(skills_pattern.findall(text)))

# Function to generate text embeddings
def get_text_embedding(texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(texts).toarray()

# Function to match resume with job description
def match_resume(resume_text, job_description):
    resume_details = extract_resume_details(resume_text)
    job_skills = extract_skills(job_description)
    resume_skills = extract_skills(resume_text)
    
    resume_summary = " ".join(resume_skills) if resume_skills else " "
    job_summary = " ".join(job_skills)
    
    embeddings = get_text_embedding([resume_summary, job_summary])
    resume_embedding, job_embedding = embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1)
    
    similarity_score = cosine_similarity(resume_embedding, job_embedding)[0][0]
    
    # Keyword matching boost
    keywords = ["html5", "css3", "javascript", "react.js", "node.js", "express.js", "mongodb", "mysql", "git", "mern", "mvc", "restful", "cloudinary", "mapbox", "multer", "bootstrap", "ejs"]
    keyword_score = sum(0.05 for keyword in keywords if keyword in resume_text.lower())
    
    # Combine TF-IDF and keyword score (cap at 1.0)
    final_score = min(1.0, similarity_score + keyword_score)
    return final_score, resume_details

# Hardcoded job description and company details
company_name = "InnovateTech Solutions"
job_description = """
Full Stack Developer (Entry-Level) at InnovateTech Solutions:
InnovateTech Solutions is seeking a motivated and enthusiastic Full Stack Developer to join our growing team. The ideal candidate will have hands-on experience with modern web development technologies and a passion for creating scalable, responsive applications. You will collaborate with cross-functional teams to design, develop, and deploy innovative web solutions, leveraging both front-end and back-end skills to deliver high-quality projects.

Responsibilities:
- Design and develop user-friendly, responsive web applications using HTML5, CSS3, JavaScript, and React.js.
- Build and maintain robust server-side applications using Node.js and Express.js.
- Manage and optimize databases with MongoDB and MySQL to support application functionality.
- Integrate and utilize version control systems (e.g., Git) for collaborative development.
- Work on full-stack projects following the MERN stack architecture.
- Participate in code reviews, testing, and debugging to ensure high-quality deliverables.
- Continuously learn and adopt new technologies to enhance project outcomes.

Requirements:
- Proficiency in front-end technologies: HTML5, CSS3, JavaScript, and React.js.
- Experience with back-end development using Node.js and Express.js.
- Familiarity with databases: MongoDB and MySQL.
- Hands-on experience with version control tools, particularly Git.
- Understanding of the MERN stack (MongoDB, Express.js, React.js, Node.js).
- Strong problem-solving skills and a proactive approach to learning.
- Ability to work independently and as part of a team.
- Pursuing or completed a degree in Computer Science, Engineering, or a related field (preferred).

Preferred Qualifications:
- Experience with personal or academic projects showcasing full-stack development (e.g., web applications).
- Knowledge of additional tools or APIs relevant to web development such as Cloudinary, MapBox, Multer, Bootstrap, and EJS.
- Demonstrated passion for continuous learning and innovation in technology.
"""

# Main execution
if __name__ == "__main__":
    # Check if the resume directory exists
    if not os.path.exists(RESUME_DIR):
        print(f"Error: Directory {RESUME_DIR} does not exist. Please check the path and try again.")
        exit(1)
    
    # Get all PDFs in the resume directory
    resume_pdfs = [f for f in os.listdir(RESUME_DIR) if f.lower().endswith(".pdf")]
    print(f"Found PDFs in {RESUME_DIR}: {resume_pdfs}")
    
    # Check if PDFs exist
    if not resume_pdfs:
        print(f"Error: No PDF files found in {RESUME_DIR}. Add your resume PDFs and try again.")
        exit(1)
    
    matched_candidates = []
    
    for pdf in resume_pdfs:
        pdf_path = os.path.join(RESUME_DIR, pdf)  # Construct full path
        resume_text = extract_text_from_pdf(pdf_path)
        if not resume_text:
            print(f"No text extracted from {pdf}")
            continue
        score, details = match_resume(resume_text, job_description)
        candidate_name = details["Name"].strip()
        # Include all candidates with score > 0.5, even with missing details
        if score > 0.5 and candidate_name.lower() != "skills":
            matched_candidates.append({
                "Email ID": details["Email"],
                "Mobile No": details["Mobile"],
                "Name": candidate_name,
                "Match Score (%)": round(score * 100, 2)
            })
    
    # Sort candidates by score (highest first)
    matched_candidates.sort(key=lambda x: x["Match Score (%)"], reverse=True)
    
    # Print job description first, then the shortlist table
    print("\nJob Description:")
    print(job_description)
    print(f"\nCompany: {company_name}")
    print("Matched Candidates:")
    if matched_candidates:
        print("+---------------------+------------------+--------------------+------------------+")
        print("| Email ID            | Mobile No        | Name               | Match Score (%)  |")
        print("+---------------------+------------------+--------------------+------------------+")
        for candidate in matched_candidates:
            email = candidate["Email ID"][:20] if len(candidate["Email ID"]) > 20 else candidate["Email ID"] + " " * (20 - len(candidate["Email ID"]))
            mobile = candidate["Mobile No"][:17] if len(candidate["Mobile No"]) > 17 else candidate["Mobile No"] + " " * (17 - len(candidate["Mobile No"]))
            name = candidate["Name"][:19] if len(candidate["Name"]) > 19 else candidate["Name"] + " " * (19 - len(candidate["Name"]))
            score = f"{candidate['Match Score (%)']:.2f}"[:18] if len(str(candidate["Match Score (%)"])) > 18 else str(candidate["Match Score (%)"]) + " " * (18 - len(str(candidate["Match Score (%)"])))
            print(f"| {email} | {mobile} | {name} | {score} |")
        print("+---------------------+------------------+--------------------+------------------+")
    else:
        print("No candidates matched the job criteria.")