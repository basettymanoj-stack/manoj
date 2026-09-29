# ============================================================
# AI STUDY HUB - COMPLETE PROJECT FOR GOOGLE COLAB
# AI GENERATIVE STUDY MATERIAL + GOOGLE FORM STYLE QUIZ
# LOGIN + SIGNUP + DASHBOARD + 5 TOPICS + TIMER + ML PREDICTION
# NO NGROK REQUIRED
# ============================================================

import os
import sqlite3
import numpy as np
import pandas as pd

from flask import Flask, request, redirect, session, jsonify, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================================
# 1. PROJECT FOLDER
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE, exist_ok=True)

DB = os.path.join(BASE, "study_hub.db")

# ============================================================
# 2. AI SCORE PREDICTION DATASET
# ============================================================

np.random.seed(42)

data = pd.DataFrame({
    "study_hours": np.random.uniform(1, 10, 500),
    "previous_score": np.random.uniform(35, 95, 500),
    "quiz_attempts": np.random.randint(1, 15, 500),
    "completion": np.random.uniform(40, 100, 500),
    "accuracy": np.random.uniform(40, 100, 500)
})

data["predicted_score"] = (
    0.25 * data["previous_score"] +
    1.8 * data["study_hours"] +
    0.8 * data["quiz_attempts"] +
    0.12 * data["completion"] +
    0.15 * data["accuracy"] +
    np.random.normal(0, 3, 500)
)

data["predicted_score"] = data["predicted_score"].clip(0, 100)

X = data[
    ["study_hours", "previous_score",
     "quiz_attempts", "completion", "accuracy"]
]

y = data["predicted_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = RandomForestRegressor(
    n_estimators=120,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("==========================================")
print("AI SCORE PREDICTION MODEL")
print("==========================================")
print("Mean Absolute Error :", round(mean_absolute_error(y_test, pred), 2))
print("R2 Score            :", round(r2_score(y_test, pred), 2))
print("==========================================")

# ============================================================
# 3. TOPICS + STUDY MATERIAL + 10 MCQs EACH
# ============================================================

topics = {

"Artificial Intelligence": {
"description": "Learn how machines simulate human intelligence.",
"material": """
Artificial Intelligence (AI) is a branch of computer science that focuses
on creating machines and software capable of performing tasks that normally
require human intelligence. These tasks include learning, reasoning,
problem solving, understanding language, recognizing images and making
decisions.

AI systems use algorithms and data to identify patterns and produce useful
results. Modern AI includes machine learning, deep learning, natural
language processing, computer vision, robotics and expert systems.

Machine learning is an important part of AI. Instead of explicitly
programming every rule, a machine learning system learns patterns from
examples. Deep learning uses neural networks containing multiple layers to
process complex information.

AI is widely used in healthcare, education, banking, transportation,
cybersecurity, recommendation systems, virtual assistants and autonomous
vehicles.


KEY CONCEPTS
AI includes machine learning, deep learning, natural language processing, computer vision, robotics and expert systems.

HOW AI WORKS
A typical AI workflow collects data, preprocesses it, selects useful features, trains a model, evaluates the model and then uses it on new inputs.

APPLICATIONS
AI is used in healthcare, education, banking, recommendation systems, virtual assistants, cybersecurity and intelligent automation.

ADVANTAGES AND CHALLENGES
AI can automate repetitive work and analyse large datasets quickly. Important challenges include data quality, bias, privacy, security and explainability.

EXAM SUMMARY
AI is the broad field; Machine Learning is a major approach within AI, and Deep Learning is based on multi-layer neural networks.""",
"questions": [
("What does AI stand for?",
 ["Artificial Intelligence","Automatic Internet","Advanced Internet","Applied Information"],0),
("Which is a branch of AI?", ["Networking","Machine Learning","Operating System","DBMS"],1),
("AI mainly attempts to simulate?", ["Human intelligence","Electricity","Storage","Networking"],0),
("Which technology uses layered neural networks?", ["Deep Learning","FTP","DNS","HTML"],0),
("Which is an AI application?", ["Virtual Assistant","Keyboard","Mouse","Printer"],0),
("AI systems learn mainly from?", ["Data","Cables","Monitors","Power"],0),
("Computer vision deals with?", ["Images and videos","Sound only","Databases only","Printers"],0),
("NLP stands for?", ["Natural Language Processing","Network Link Protocol","New Learning Program","Natural Logic Process"],0),
("Robotics combines AI with?", ["Machines","HTML","SQL","Email"],0),
("Which is an AI goal?", ["Decision making","Increasing cable length","Formatting disks","Printing pages"],0)
]
},

"Machine Learning": {
"description": "Understand algorithms that learn patterns from data.",
"material": """
Machine Learning (ML) is a subset of Artificial Intelligence that enables
computers to learn from data without being explicitly programmed for every
task. A machine learning model identifies patterns in training data and
uses those patterns to make predictions or decisions.

The major types of machine learning are supervised learning, unsupervised
learning and reinforcement learning. Supervised learning uses labelled
data. Classification predicts categories, while regression predicts
continuous numerical values.

Unsupervised learning works with unlabelled data and discovers hidden
patterns. Clustering is a common example. Reinforcement learning learns
through rewards and penalties while interacting with an environment.

Popular machine learning algorithms include Linear Regression, Logistic
Regression, Decision Trees, Random Forest, Support Vector Machines,
K-Nearest Neighbours and K-Means.


KEY CONCEPTS
Machine learning includes supervised, unsupervised and reinforcement learning. Classification predicts categories, regression predicts numerical values and clustering discovers groups.

WORKFLOW
A common workflow is data collection, cleaning, feature preparation, train-test splitting, model training, evaluation and deployment.

APPLICATIONS
ML is used for spam filtering, recommendation systems, fraud detection, medical prediction, image recognition and customer analytics.

ADVANTAGES AND CHALLENGES
ML can discover complex patterns, but poor data, overfitting, underfitting and bias can reduce model quality.

EXAM SUMMARY
A good model should generalize to unseen data instead of simply memorizing training examples.""",
"questions": [
("Machine Learning is a subset of?", ["AI","DBMS","OS","Networking"],0),
("Supervised learning uses?", ["Labelled data","No data","Only images","Only audio"],0),
("Classification predicts?", ["Categories","Continuous values only","Memory","Network speed"],0),
("Regression predicts?", ["Numerical values","Passwords","Files","Ports"],0),
("K-Means is used for?", ["Clustering","Encryption","Routing","Compilation"],0),
("Random Forest is based on?", ["Decision Trees","DNS","TCP","HTML"],0),
("KNN stands for?", ["K-Nearest Neighbours","Key Network Node","Kernel New Network","Known Neural Node"],0),
("Reinforcement learning uses?", ["Rewards and penalties","SQL tables","Routers","Compilers"],0),
("Training data is used to?", ["Train a model","Create hardware","Connect cables","Install OS"],0),
("A model makes predictions using?", ["Learned patterns","Keyboard","Monitor","Printer"],0)
]
},

"Data Science": {
"description": "Learn how data is collected, processed, analysed and visualized.",
"material": """
Data Science is an interdisciplinary field that combines statistics,
programming, mathematics and domain knowledge to extract useful information
from data. It helps organizations make better decisions using evidence
obtained from datasets.

The data science process usually includes data collection, data cleaning,
preprocessing, exploratory data analysis, visualization, feature
engineering, model building and evaluation.

Python is one of the most popular languages for data science. Pandas is
used for data manipulation, NumPy provides numerical operations and
Matplotlib is used for visualization. Scikit-learn provides many machine
learning algorithms.

Data science is used in business analytics, healthcare, finance,
education, marketing, recommendation systems and scientific research.


KEY CONCEPTS
Data Science combines statistics, programming, mathematics, visualization and domain knowledge.

WORKFLOW
The main stages are data collection, cleaning, exploratory analysis, feature engineering, modelling, evaluation and communication.

APPLICATIONS
It supports healthcare analysis, financial forecasting, customer segmentation, scientific research and business analytics.

ADVANTAGES AND CHALLENGES
Data Science helps organizations make evidence-based decisions. Data quality, privacy, bias and large data volumes remain important challenges.

EXAM SUMMARY
Remember: Collect → Clean → Explore → Model → Evaluate → Communicate.""",
"questions": [
("Data Science combines?", ["Statistics and programming","Only hardware","Only networking","Only HTML"],0),
("Which Python library handles tables?", ["Pandas","Flask","TensorFlow only","OS"],0),
("NumPy is mainly used for?", ["Numerical computing","Email","Networking","Web hosting"],0),
("Matplotlib is used for?", ["Visualization","Encryption","Routing","Compilation"],0),
("Data cleaning removes?", ["Errors and inconsistencies","Monitors","Routers","Passwords"],0),
("EDA stands for?", ["Exploratory Data Analysis","Electronic Data Access","External Database Algorithm","Easy Data Application"],0),
("Feature engineering creates?", ["Useful input features","Routers","Websites","Emails"],0),
("A dataset contains?", ["Data records","Only programs","Only cables","Only images"],0),
("Data visualization represents data using?", ["Charts and graphs","Passwords","Routers","Compilers"],0),
("Scikit-learn is used for?", ["Machine Learning","Video editing","Networking only","Operating systems"],0)
]
},

"Computer Networks": {
"description": "Study communication between computers and network devices.",
"material": """
A computer network is a collection of interconnected devices that
communicate and share resources. Networks allow computers, servers,
smartphones and other devices to exchange data.

The OSI model contains seven layers: Physical, Data Link, Network,
Transport, Session, Presentation and Application. The TCP/IP model is
widely used in real-world networks.

Important protocols include TCP, UDP, IP, HTTP, HTTPS, DNS, FTP and SMTP.
TCP provides reliable and ordered delivery, while UDP provides faster
connectionless communication.

Network devices include switches, routers, hubs, bridges, gateways and
access points. Routing determines the path through which packets travel
from source to destination.


KEY CONCEPTS
Networking uses layered models such as OSI and TCP/IP. Protocols define how devices exchange information.

HOW COMMUNICATION WORKS
Applications create data, transport protocols provide end-to-end communication, network protocols handle logical addressing and routing, and lower layers deliver frames and signals.

APPLICATIONS
Networks support websites, email, cloud services, video conferencing, gaming and file transfer.

ADVANTAGES AND CHALLENGES
Networks enable resource sharing and long-distance communication. Congestion, failures, latency and security attacks are important challenges.

EXAM SUMMARY
Switches mainly forward frames using MAC addresses; routers forward packets between networks using IP addresses and routing information.""",
"questions": [
("How many layers are in OSI?", ["7","4","5","6"],0),
("Which layer handles routing?", ["Network","Physical","Presentation","Session"],0),
("TCP provides?", ["Reliable delivery","No delivery","Only encryption","Only routing"],0),
("UDP is?", ["Connectionless","Connection-oriented","A database","An OS"],0),
("DNS converts?", ["Domain names to IP addresses","IP to passwords","Files to folders","HTML to CSS"],0),
("HTTP is used for?", ["Web communication","Routing only","Printing","Compiling"],0),
("A router connects?", ["Networks","Only keyboards","Only monitors","Only files"],0),
("IP is responsible for?", ["Addressing and routing","Typing","Printing","Video editing"],0),
("SMTP is used for?", ["Email","Web pages","Routing","Encryption only"],0),
("A switch mainly connects?", ["Devices in a LAN","Different countries","Only servers","Only websites"],0)
]
},

"Cyber Security": {
"description": "Learn how systems, networks and data are protected.",
"material": """
Cyber Security is the practice of protecting computers, networks,
applications and data from unauthorized access, attacks and damage.
Security is important because modern organizations depend heavily on
digital systems.

The three main security goals are confidentiality, integrity and
availability, commonly called the CIA triad. Confidentiality prevents
unauthorized disclosure, integrity protects information from improper
modification and availability ensures authorized users can access systems.

Common threats include malware, phishing, ransomware, password attacks,
SQL injection, cross-site scripting and denial-of-service attacks.
Security controls include firewalls, antivirus software, encryption,
multi-factor authentication, access control and regular backups.

Ethical hackers test systems with permission to identify vulnerabilities
before attackers can exploit them.


KEY CONCEPTS
Cybersecurity protects confidentiality, integrity and availability using authentication, authorization, access control, encryption, firewalls, backups and monitoring.

SECURITY WORKFLOW
Security teams identify assets, assess risks, protect systems, monitor activity, respond to incidents and recover services.

APPLICATIONS
Cybersecurity is essential in banking, healthcare, education, cloud computing, e-commerce and government systems.

ADVANTAGES AND CHALLENGES
Security controls reduce unauthorized access and data loss. Phishing, malware, vulnerabilities, credential theft and human error remain major challenges.

EXAM SUMMARY
CIA means Confidentiality, Integrity and Availability.""",
"questions": [
("CIA stands for?", ["Confidentiality Integrity Availability","Computer Internet Access","Central Information Application","Cyber Internal Architecture"],0),
("Malware means?", ["Malicious software","Network cable","Database","Browser"],0),
("Phishing attempts to steal?", ["Sensitive information","Electricity","Hardware","Printers"],0),
("Firewall helps to?", ["Control network traffic","Create websites","Edit videos","Compile programs"],0),
("Encryption protects?", ["Data","Keyboard","Monitor","Printer"],0),
("SQL Injection targets?", ["Databases/applications","Routers only","Monitors","Operating hardware"],0),
("XSS stands for?", ["Cross-Site Scripting","Extra Secure System","Cross Server Security","External System Software"],0),
("MFA means?", ["Multi-Factor Authentication","Main File Access","Multiple Firewall Application","Managed File Algorithm"],0),
("Ethical hackers work with?", ["Permission","No permission","Only hardware","Only printers"],0),
("Backup helps against?", ["Data loss","Keyboard failure only","Screen brightness","Network cables"],0)
]
}

}


# ---------------- ADDITIONAL STUDY TOPICS ----------------

topics.update({

"Python Programming": {
"description": "Learn Python basics, functions, collections, OOP and practical programming.",
"material": """
Python is a high-level, interpreted programming language known for its simple
syntax and readability. It is widely used in web development, automation,
data science, artificial intelligence, machine learning and software development.

Python provides variables, operators, conditional statements, loops, functions,
lists, tuples, sets and dictionaries. Functions help divide a large program into
smaller reusable parts. Python also supports object-oriented programming using
classes and objects.

Libraries extend Python capabilities. NumPy supports numerical computing,
Pandas supports data analysis, Matplotlib supports visualization and Scikit-learn
provides machine learning algorithms. Python is popular because programs can be
developed quickly with comparatively less code.


KEY CONCEPTS
Python supports variables, conditions, loops, functions, collections, modules, exceptions and object-oriented programming.

WORKFLOW
Programs can be divided into reusable functions and modules. Input is processed, logic is executed and output is produced. Exceptions can be handled using try and except.

APPLICATIONS
Python is used for automation, web development, data analysis, artificial intelligence, machine learning and scripting.

ADVANTAGES AND CHALLENGES
Python is readable and has a large library ecosystem. Some CPU-intensive workloads may run slower than compiled languages.

EXAM SUMMARY
List = ordered and mutable, Tuple = ordered and immutable, Set = unique values, Dictionary = key-value pairs.""",
"questions": [
("Python is a?", ["Programming language","Database","Network device","Operating system"],0),
("Which symbol starts a Python comment?", ["#","//","<!--","**"],0),
("Which collection stores key-value pairs?", ["Dictionary","List","Tuple","String"],0),
("Which keyword defines a function?", ["def","func","function","define"],0),
("Which library is used for numerical arrays?", ["NumPy","Flask","SQLite","HTML"],0),
("Which library is commonly used for data analysis?", ["Pandas","FTP","DNS","SMTP"],0),
("Which statement is used for repetition?", ["for","import","class","return"],0),
("Which keyword creates a class?", ["class","object","struct","create"],0),
("Python uses indentation for?", ["Code blocks","Encryption","Routing","Storage"],0),
("Which is a Python IDE/notebook environment?", ["Jupyter","DNS","TCP","ARP"],0)
]},

"Database Management Systems": {
"description": "Study databases, SQL, transactions, normalization and database design.",
"material": """
A Database Management System (DBMS) is software used to create, store, organize
and retrieve data efficiently. It provides controlled access to data and supports
operations such as insertion, deletion, updating and querying.

A relational database stores information in tables containing rows and columns.
SQL is used to define, manipulate and retrieve relational data. Important SQL
commands include SELECT, INSERT, UPDATE and DELETE.

Database design uses concepts such as keys, relationships and normalization.
Transactions maintain consistency using properties called ACID: atomicity,
consistency, isolation and durability. Concurrency control and recovery help
protect databases when multiple users access data or failures occur.


KEY CONCEPTS
A DBMS manages tables, relationships, keys, queries, transactions and concurrency. Primary keys uniquely identify rows and foreign keys connect related tables.

SQL AND TRANSACTIONS
SELECT retrieves data, INSERT adds records, UPDATE changes records and DELETE removes records. ACID properties help maintain reliable transactions.

APPLICATIONS
DBMS technology is used in banking, hospitals, colleges, reservation systems, e-commerce and enterprise software.

ADVANTAGES AND CHALLENGES
A DBMS provides organized storage, controlled access and reliable querying. Security, concurrency, backup and recovery must be managed carefully.

EXAM SUMMARY
ACID means Atomicity, Consistency, Isolation and Durability.""",
"questions": [
("DBMS stands for?", ["Database Management System","Data Binary Management Software","Database Machine System","Digital Base Management Service"],0),
("A relational database stores data in?", ["Tables","Routers","Files only","Images only"],0),
("SQL is used to?", ["Manage/query databases","Route packets","Compile Java","Design hardware"],0),
("Which command retrieves data?", ["SELECT","DELETE","DROP","UPDATE"],0),
("Which command adds records?", ["INSERT","SELECT","ROUTE","PRINT"],0),
("A primary key identifies?", ["A row uniquely","A network","A program","A monitor"],0),
("ACID is related to?", ["Transactions","Web design","Routing","Encryption only"],0),
("Normalization helps reduce?", ["Data redundancy","Network speed","Screen size","CPU frequency"],0),
("Which command modifies existing data?", ["UPDATE","SELECT","PING","TRACE"],0),
("Which property means all-or-nothing?", ["Atomicity","Isolation","Durability","Availability"],0)
]},

"Operating Systems": {
"description": "Understand processes, memory, scheduling, files and operating-system services.",
"material": """
An Operating System (OS) is system software that manages computer hardware
and provides services for application programs. It acts as an interface between
users, applications and hardware.

Major operating-system functions include process management, memory management,
file management, device management and security. A process is a program in
execution. Operating systems schedule processes using algorithms such as FCFS,
SJF, Priority and Round Robin.

Memory management allocates and deallocates memory. Virtual memory allows a
system to use secondary storage to extend available logical memory. File systems
organize files and directories, while protection mechanisms control access to
system resources.


KEY CONCEPTS
An operating system manages processes, memory, files, devices and security while providing services to applications.

PROCESS AND MEMORY MANAGEMENT
CPU scheduling selects processes for execution. Memory management allocates and protects memory. Virtual memory can use secondary storage when required.

APPLICATIONS
Operating-system concepts are used in desktops, servers, smartphones, embedded systems and cloud platforms.

ADVANTAGES AND CHALLENGES
An OS provides resource management and security. Deadlocks, memory pressure, scheduling conflicts and vulnerabilities are important challenges.

EXAM SUMMARY
Major responsibilities include Process Management, Memory Management, File Management, Device Management and Security.""",
"questions": [
("OS stands for?", ["Operating System","Open Software","Output System","Online Service"],0),
("An OS manages?", ["Hardware and software resources","Only websites","Only databases","Only printers"],0),
("A program in execution is a?", ["Process","File","Packet","Thread only"],0),
("Round Robin is a?", ["CPU scheduling algorithm","Database command","Network protocol","File system"],0),
("Virtual memory uses?", ["Secondary storage as an extension of memory","Only cache","Only ROM","Only CPU registers"],0),
("Which manages files?", ["File system","Router","Compiler","Browser"],0),
("FCFS stands for?", ["First Come First Served","Fast CPU File Service","First Control File System","File Control First Service"],0),
("Which manages processes?", ["Process management","DNS","HTML","SQL"],0),
("Memory allocation belongs to?", ["Memory management","Email","Routing","Web design"],0),
("OS provides an interface between?", ["User/applications and hardware","Only routers","Only databases","Only websites"],0)
]},

"Cloud Computing": {
"description": "Learn cloud models, virtualization, services, deployment and cloud architecture.",
"material": """
Cloud Computing provides computing resources such as servers, storage,
networking, platforms and software through network-based services. Users can
obtain resources when needed instead of maintaining all physical infrastructure.

The major cloud service models are Infrastructure as a Service (IaaS),
Platform as a Service (PaaS) and Software as a Service (SaaS). Deployment
models include public cloud, private cloud, hybrid cloud and community cloud.

Virtualization is an important cloud technology. A hypervisor creates and
manages virtual machines on physical hardware. Cloud systems support scalability,
resource pooling, elasticity and measured service. Examples of cloud platforms
include AWS, Microsoft Azure and Google Cloud.


KEY CONCEPTS
Cloud computing provides resources on demand. IaaS provides infrastructure, PaaS provides a development platform and SaaS provides ready-to-use software.

HOW CLOUD WORKS
Virtualized or containerized resources are pooled in data centers. Users request resources through cloud platforms and resources are allocated according to demand.

APPLICATIONS
Cloud services support web hosting, backups, analytics, software development, databases and machine-learning workloads.

ADVANTAGES AND CHALLENGES
Cloud computing provides scalability and flexibility. Security, privacy, availability, vendor lock-in and cost control are important concerns.

EXAM SUMMARY
Remember IaaS = infrastructure, PaaS = platform, SaaS = software.""",
"questions": [
("Cloud computing provides?", ["Computing resources as services","Only cables","Only printers","Only keyboards"],0),
("IaaS stands for?", ["Infrastructure as a Service","Internet as a System","Information as Software","Integrated Application Service"],0),
("SaaS provides?", ["Software as a service","Only hardware","Only storage chips","Network cables"],0),
("PaaS stands for?", ["Platform as a Service","Program as a System","Packet as a Service","Private Application System"],0),
("A hypervisor manages?", ["Virtual machines","Emails","Web pages","SQL tables"],0),
("Public cloud is?", ["Cloud infrastructure available to multiple customers","Only local hardware","A programming language","A database"],0),
("Hybrid cloud combines?", ["Private and public cloud environments","CPU and keyboard","SQL and HTML","LAN and printer"],0),
("Cloud elasticity means?", ["Resources can scale according to demand","Files are encrypted only","Routers are replaced","Screens resize"],0),
("AWS is a?", ["Cloud platform","Programming language","Database command","Network cable"],0),
("Virtualization creates?", ["Virtual computing environments","Email accounts","Passwords","Web browsers"],0)
]},

"Computer Science Fundamentals": {
"description": "Revise algorithms, data structures, programming and core computing concepts.",
"material": """
Computer science fundamentals provide the foundation for software development.
Algorithms are step-by-step procedures used to solve problems. A good algorithm
should be clear, finite and produce the required result.

Data structures organize data for efficient processing. Arrays, linked lists,
stacks, queues, trees and graphs are common structures. Searching and sorting
algorithms are used frequently in applications.

Time and space complexity describe the resources required by an algorithm.
Big-O notation is commonly used to express how running time grows with input
size. Understanding these concepts helps developers select suitable solutions.


KEY CONCEPTS
Core computer science includes algorithms, data structures, programming, complexity analysis, databases, operating systems and networking.

ALGORITHMS AND DATA STRUCTURES
Arrays, linked lists, stacks, queues, trees and graphs organize data for efficient processing.

COMPLEXITY
Big-O notation describes how time or space requirements grow as input size increases.

APPLICATIONS
These fundamentals are used in software engineering, compiler design, databases, operating systems, AI and competitive programming.

EXAM SUMMARY
Stack follows LIFO, Queue follows FIFO, and binary search requires sorted data.""",
"questions": [
("An algorithm is a?", ["Step-by-step problem-solving procedure","Database","Network","Hardware device"],0),
("A stack follows?", ["LIFO","FIFO","Random only","Priority only"],0),
("A queue follows?", ["FIFO","LIFO","Binary only","Random only"],0),
("A tree is a?", ["Non-linear data structure","Network cable","Database command","Compiler"],0),
("Graphs contain?", ["Vertices and edges","Rows only","Columns only","Passwords"],0),
("Big-O describes?", ["Growth of resource usage","Screen size","Network address","File name"],0),
("Binary search requires data to be?", ["Sorted","Encrypted","Printed","Compressed"],0),
("A linked list contains?", ["Nodes linked together","Only arrays","Only files","Only packets"],0),
("Which is a sorting algorithm?", ["Merge Sort","DNS","HTTP","ARP"],0),
("An array usually stores?", ["Elements in indexed positions","Only passwords","Only routers","Only web pages"],0)
]},

"Web Development": {
"description": "Learn HTML, CSS, JavaScript, client-server concepts and modern web applications.",
"material": """
Web development involves creating websites and web applications. HTML defines
the structure of a web page, CSS controls presentation and JavaScript adds
interactivity and dynamic behavior.

A web application commonly uses a client-server architecture. The browser acts
as a client and communicates with a server using HTTP or HTTPS. Backend systems
can process requests, communicate with databases and return responses.

Modern development can use frameworks and libraries for reusable components.
Responsive design helps websites adapt to different screen sizes. Security
practices such as input validation, authentication and secure communication are
important when building web applications.


KEY CONCEPTS
Web development includes frontend and backend components. HTML defines structure, CSS controls presentation and JavaScript adds interaction.

WEB APPLICATION FLOW
A browser sends an HTTP request to a server. The backend processes the request, may access a database and returns an HTTP response. APIs often exchange JSON data.

APPLICATIONS
Web technologies are used for e-commerce, education portals, banking, dashboards, social platforms and enterprise applications.

SECURITY
Applications should validate input, protect sessions, use secure authentication and apply safe database access. Injection and cross-site scripting are common concerns.

EXAM SUMMARY
Frontend focuses on the user interface; backend focuses on server logic, data access and application services.""",
"questions": [
("HTML is used for?", ["Page structure","Database storage","Routing","Encryption"],0),
("CSS is used for?", ["Styling","Database queries","Packet routing","Password hashing"],0),
("JavaScript adds?", ["Interactivity","Physical memory","Network cables","Database hardware"],0),
("HTTP is a?", ["Web communication protocol","Programming language","Database","OS"],0),
("HTTPS provides?", ["Secure HTTP communication","Faster CPU","More RAM","File compression only"],0),
("A browser is commonly a?", ["Client","Database","Router","Compiler"],0),
("Backend can communicate with?", ["Databases","Only monitors","Only keyboards","Only printers"],0),
("Responsive design adapts to?", ["Different screen sizes","Only passwords","Only databases","Only routers"],0),
("Input validation helps?", ["Check submitted data","Increase RAM","Change CPU","Create cables"],0),
("HTML stands for?", ["HyperText Markup Language","High Transfer Machine Language","Hyperlink Text Management Logic","Home Tool Markup Language"],0)
]}
})


# ============================================================
# 4. DATABASE
# ============================================================

conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS results(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    topic TEXT,
    score INTEGER,
    total INTEGER,
    percentage REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("Database created successfully.")

# ============================================================
# 5. FLASK APP
# ============================================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "AI_STUDY_HUB_SECRET_2026")

# ============================================================
# 6. COMMON CSS
# ============================================================

CSS = """

*{
    box-sizing:border-box;
    margin:0;
    padding:0;
    font-family:Arial, sans-serif;
}

body{
    min-height:100vh;
    color:white;
    background:
    radial-gradient(circle at 20% 20%, #7c3aed, transparent 30%),
    radial-gradient(circle at 80% 10%, #06b6d4, transparent 30%),
    radial-gradient(circle at 50% 90%, #ec4899, transparent 30%),
    linear-gradient(135deg,#020617,#111827,#1e1b4b);
    background-attachment:fixed;
}

a{
    text-decoration:none;
    color:inherit;
}

.container{
    width:92%;
    max-width:1200px;
    margin:auto;
}

.navbar{
    padding:20px 0;
    display:flex;
    justify-content:space-between;
    align-items:center;
}

.logo{
    font-size:28px;
    font-weight:bold;
}

.logo span{
    color:#22d3ee;
}

.btn{
    border:none;
    padding:13px 22px;
    border-radius:12px;
    cursor:pointer;
    font-weight:bold;
    color:white;
    background:linear-gradient(135deg,#7c3aed,#06b6d4);
    transition:.3s;
}

.btn:hover{
    transform:translateY(-2px);
    box-shadow:0 10px 30px #0005;
}

.card{
    background:#ffffff12;
    border:1px solid #ffffff20;
    backdrop-filter:blur(20px);
    border-radius:22px;
    padding:25px;
    box-shadow:0 20px 50px #0005;
}

.center{
    min-height:100vh;
    display:flex;
    justify-content:center;
    align-items:center;
    padding:25px;
}

.auth{
    width:430px;
}

.auth h1{
    text-align:center;
    margin-bottom:10px;
}

.auth p{
    text-align:center;
    color:#cbd5e1;
    margin-bottom:25px;
}

input{
    width:100%;
    padding:15px;
    margin:8px 0 15px;
    border:none;
    outline:none;
    border-radius:12px;
    background:#ffffff15;
    color:white;
    border:1px solid #ffffff25;
}

input::placeholder{
    color:#cbd5e1;
}

.form-btn{
    width:100%;
}

.link{
    color:#67e8f9;
}

.hero{
    padding:45px 0;
}

.hero h1{
    font-size:46px;
    margin-bottom:15px;
}

.hero p{
    color:#cbd5e1;
    max-width:800px;
    line-height:1.8;
}

.grid{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
    gap:20px;
    margin:30px 0;
}

.topic-card{
    min-height:240px;
    transition:.3s;
}

.topic-card:hover{
    transform:translateY(-8px);
}

.topic-icon{
    font-size:45px;
    margin-bottom:15px;
}

.topic-card h2{
    margin-bottom:12px;
}

.topic-card p{
    color:#cbd5e1;
    line-height:1.6;
    margin-bottom:20px;
}

.stats{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:15px;
    margin:25px 0;
}

.stat{
    text-align:center;
}

.stat h2{
    font-size:32px;
    color:#67e8f9;
}

.stat p{
    color:#cbd5e1;
}

.study{
    line-height:1.9;
    color:#e2e8f0;
    font-size:17px;
}

.study p{
    margin-bottom:20px;
}

.question{
    margin-bottom:25px;
}

.question h3{
    margin-bottom:15px;
}

.option{
    display:block;
    padding:14px;
    margin:9px 0;
    border-radius:12px;
    background:#ffffff0d;
    border:1px solid #ffffff18;
    cursor:pointer;
}

.option:hover{
    background:#ffffff18;
}

.timer{
    position:sticky;
    top:15px;
    z-index:10;
    text-align:center;
    padding:15px;
    margin-bottom:20px;
    border-radius:15px;
    background:#ef4444dd;
    font-size:22px;
    font-weight:bold;
}

.result{
    text-align:center;
    margin-top:20px;
}

.big-score{
    font-size:65px;
    color:#67e8f9;
    font-weight:bold;
}

table{
    width:100%;
    border-collapse:collapse;
    margin-top:20px;
}

th,td{
    padding:13px;
    border-bottom:1px solid #ffffff20;
    text-align:left;
}

th{
    color:#67e8f9;
}

@media(max-width:700px){
    .hero h1{
        font-size:32px;
    }

    .stats{
        grid-template-columns:repeat(2,1fr);
    }

    .navbar{
        gap:10px;
    }
}

"""

# ============================================================
# 7. LOGIN PAGE
# ============================================================

LOGIN = """
<!DOCTYPE html>
<html>
<head>
<title>AI Study Hub - Login</title>
<style>{{css}}
.auth-modern{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:30px;position:relative;overflow:hidden}
.auth-modern:before,.auth-modern:after{content:"";position:absolute;border-radius:50%;filter:blur(75px);opacity:.45;animation:float 8s ease-in-out infinite}
.auth-modern:before{width:300px;height:300px;background:#8b5cf6;top:4%;left:4%}
.auth-modern:after{width:320px;height:320px;background:#06b6d4;bottom:0;right:4%;animation-delay:2s}
.auth-box{position:relative;z-index:2;width:100%;max-width:460px;padding:42px;border:1px solid rgba(255,255,255,.18);border-radius:30px;background:rgba(15,23,42,.68);backdrop-filter:blur(25px);box-shadow:0 30px 90px rgba(0,0,0,.42)}
.auth-logo{width:82px;height:82px;margin:0 auto 18px;border-radius:25px;display:flex;align-items:center;justify-content:center;font-size:44px;background:linear-gradient(135deg,#8b5cf6,#06b6d4);box-shadow:0 0 42px rgba(6,182,212,.35)}
.auth-title{text-align:center;font-size:35px;margin:8px 0}.auth-sub{text-align:center;color:#cbd5e1;margin-bottom:28px}
.auth-error{padding:12px 15px;border-radius:12px;background:rgba(239,68,68,.18);color:#fecaca;margin-bottom:15px}
.pass-wrap{position:relative}.pass-wrap input{padding-right:50px}.eye{position:absolute;right:8px;top:8px;border:0;background:transparent;color:white;font-size:18px;cursor:pointer}
@keyframes float{50%{transform:translateY(-24px) scale(1.05)}}
</style>
</head>
<body>
<div class="auth-modern">
<div class="auth-box">
<div class="auth-logo">🎓</div>
<h1 class="auth-title">AI Study Hub</h1>
<p class="auth-sub">Learn smarter. Practice better. Grow faster.</p>
{% if error %}<div class="auth-error">{{error}}</div>{% endif %}
<form method="POST">
<input type="email" name="email" placeholder="Email address" autocomplete="username" required>
<div class="pass-wrap">
<input id="loginPassword" type="password" name="password" placeholder="Password" autocomplete="current-password" required>
<button class="eye" type="button" onclick="togglePassword('loginPassword',this)">👁️</button>
</div>
<button class="btn form-btn" style="width:100%;margin-top:10px">🚀 Sign In</button>
</form>
<p style="text-align:center;color:#cbd5e1;margin-top:24px">New to AI Study Hub?</p>
<div style="text-align:center"><a class="btn" href="/signup">✨ Create Account</a></div>
</div>
</div>
<script>
function togglePassword(id,btn){const x=document.getElementById(id);if(x.type==="password"){x.type="text";btn.textContent="🙈"}else{x.type="password";btn.textContent="👁️"}}
</script>
</body>
</html>
"""

# ============================================================
# 8. SIGNUP PAGE
# ============================================================

SIGNUP = """
<!DOCTYPE html>
<html>
<head>
<title>AI Study Hub - Sign Up</title>
<style>{{css}}
.auth-modern{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:30px;position:relative;overflow:hidden}
.auth-modern:before,.auth-modern:after{content:"";position:absolute;border-radius:50%;filter:blur(75px);opacity:.45;animation:float 8s ease-in-out infinite}
.auth-modern:before{width:300px;height:300px;background:#ec4899;top:3%;left:4%}
.auth-modern:after{width:320px;height:320px;background:#06b6d4;bottom:0;right:4%;animation-delay:2s}
.auth-box{position:relative;z-index:2;width:100%;max-width:460px;padding:42px;border:1px solid rgba(255,255,255,.18);border-radius:30px;background:rgba(15,23,42,.68);backdrop-filter:blur(25px);box-shadow:0 30px 90px rgba(0,0,0,.42)}
.auth-logo{width:82px;height:82px;margin:0 auto 18px;border-radius:25px;display:flex;align-items:center;justify-content:center;font-size:44px;background:linear-gradient(135deg,#ec4899,#8b5cf6,#06b6d4);box-shadow:0 0 42px rgba(236,72,153,.35)}
.auth-title{text-align:center;font-size:35px;margin:8px 0}.auth-sub{text-align:center;color:#cbd5e1;margin-bottom:28px}
.auth-error{padding:12px 15px;border-radius:12px;background:rgba(239,68,68,.18);color:#fecaca;margin-bottom:15px}
.pass-wrap{position:relative}.pass-wrap input{padding-right:50px}.eye{position:absolute;right:8px;top:8px;border:0;background:transparent;color:white;font-size:18px;cursor:pointer}
@keyframes float{50%{transform:translateY(-24px) scale(1.05)}}
</style>
</head>
<body>
<div class="auth-modern">
<div class="auth-box">
<div class="auth-logo">✨</div>
<h1 class="auth-title">Create Your Account</h1>
<p class="auth-sub">Join your personalized AI learning space.</p>
{% if error %}<div class="auth-error">{{error}}</div>{% endif %}
<form method="POST">
<input type="text" name="name" placeholder="Full name" autocomplete="name" required>
<input type="email" name="email" placeholder="Email address" autocomplete="email" required>
<div class="pass-wrap">
<input id="signupPassword" type="password" name="password" placeholder="Create password" autocomplete="new-password" required>
<button class="eye" type="button" onclick="togglePassword('signupPassword',this)">👁️</button>
</div>
<button class="btn form-btn" style="width:100%;margin-top:10px">🚀 Create Account</button>
</form>
<p style="text-align:center;color:#cbd5e1;margin-top:24px">Already have an account?</p>
<div style="text-align:center"><a class="btn" href="/login">🔐 Login</a></div>
</div>
</div>
<script>
function togglePassword(id,btn){const x=document.getElementById(id);if(x.type==="password"){x.type="text";btn.textContent="🙈"}else{x.type="password";btn.textContent="👁️"}}
</script>
</body>
</html>
"""

# ============================================================
# 9. DASHBOARD
# ============================================================

DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
<title>AI Study Hub Dashboard</title>
<style>{{css}}</style>
</head>

<body>

<div class="container">

<div class="navbar">

<div class="logo">
🤖 AI <span>STUDY HUB</span>
</div>

<div>
<span>Hi, {{name}} 👋</span>
&nbsp;&nbsp;
<a class="btn" href="/logout">Logout</a>
</div>

</div>

<section class="hero">

<h1>Learn Smarter with AI 🚀</h1>

<p>
Welcome to your AI-powered learning dashboard.
Read study materials, attend timed quizzes, track your performance
and predict your future score using machine learning.
</p>

</section>

<div class="stats">

<div class="card stat">
<h2>{{total_quizzes}}</h2>
<p>Quiz Attempts</p>
</div>

<div class="card stat">
<h2>{{avg_score}}%</h2>
<p>Average Score</p>
</div>

<div class="card stat">
<h2>{{best_score}}%</h2>
<p>Best Score</p>
</div>

<div class="card stat">
<h2>{{topics|length}}</h2>
<p>Topics</p>
</div>

</div>

<div class="grid">

{% for topic in topics %}

<div class="card topic-card">

<div class="topic-icon">
{{icons[loop.index0]}}
</div>

<h2>{{topic}}</h2>

<p>{{descriptions[topic]}}</p>

<a class="btn" href="/study/{{topic}}">
Study Material
</a>

<a class="btn" href="/quiz/{{topic}}">
Quiz
</a>

</div>

{% endfor %}

</div>

<div class="card" style="margin-bottom:30px">

<h2>🚀 Learning Center</h2>
<p style="color:#cbd5e1;line-height:1.7;margin:12px 0 20px">
Explore all study materials, complete 10-question quizzes for every topic,
review your performance and use AI score prediction to plan your learning.
</p>

<a class="btn" href="/prediction">
📊 Predict My Future Score
</a>

</div>

</div>


<style>
body.color-cycle{
    transition: background 1.2s ease, background-color 1.2s ease;
}
@keyframes hueShift {
    0% { filter:hue-rotate(0deg); }
    25% { filter:hue-rotate(70deg); }
    50% { filter:hue-rotate(150deg); }
    75% { filter:hue-rotate(240deg); }
    100% { filter:hue-rotate(360deg); }
}
.color-cycle .card{
    transition:box-shadow 1.2s ease, border-color 1.2s ease;
}
.color-cycle .btn{
    transition:background 1.2s ease, transform .3s ease;
}
</style>
<script>
(function(){
    const colors = [
        ["#020617","#312e81","#0e7490","#9d174d"],
        ["#06121a","#064e3b","#155e75","#7e22ce"],
        ["#170b2e","#581c87","#0f766e","#be123c"],
        ["#111827","#1d4ed8","#0f766e","#a21caf"],
        ["#0f172a","#7c2d12","#166534","#4338ca"],
        ["#172554","#075985","#701a75","#9f1239"]
    ];
    let i = 0;
    function changeColors(){
        const c = colors[i % colors.length];
        document.body.style.background =
          `radial-gradient(circle at 18% 18%, ${c[1]}, transparent 32%),
           radial-gradient(circle at 82% 12%, ${c[2]}, transparent 32%),
           radial-gradient(circle at 52% 90%, ${c[3]}, transparent 35%),
           linear-gradient(135deg,${c[0]},#111827,#1e1b4b)`;
        i++;
    }
    document.body.classList.add("color-cycle");
    changeColors();
    setInterval(changeColors, 5000);
})();
</script>

</body>
</html>
"""

# ============================================================
# 10. STUDY PAGE
# ============================================================

STUDY = """
<!DOCTYPE html>
<html>
<head>
<title>{{topic}} - Study Material</title>
<style>{{css}}</style>
</head>

<body>

<div class="container">

<div class="navbar">

<div class="logo">
🤖 AI <span>STUDY HUB</span>
</div>

<a class="btn" href="/dashboard">
Dashboard
</a>

</div>

<div class="card">

<h1>📚 {{topic}}</h1>

<br>

<div class="study">

{% for paragraph in material.split('\\n') %}

{% if paragraph.strip() %}

<p>{{paragraph}}</p>

{% endif %}

{% endfor %}

</div>

<br>

<a class="btn" href="/quiz/{{topic}}">
Start 5 Minute Quiz →
</a>

</div>

</div>

</body>
</html>
"""

# ============================================================
# 11. QUIZ PAGE
# ============================================================

QUIZ = """
<!DOCTYPE html>
<html>
<head>
<title>{{topic}} Quiz</title>
<style>{{css}}</style>
</head>

<body>

<div class="container">

<div class="navbar">

<div class="logo">
🤖 AI <span>QUIZ</span>
</div>

<a class="btn" href="/dashboard">
Dashboard
</a>

</div>

<div class="timer" id="timer">
⏱️ Time Left: 05:00
</div>

<div class="card">

<h1>📝 {{topic}} Quiz</h1>

<p style="color:#cbd5e1;margin-top:10px">
10 Questions • 5 Minutes • Google Form Style
</p>

<form id="quizForm">

{% for q in questions %}
{% set q_index = loop.index0 %}

<div class="question">

<h3>
{{loop.index}}. {{q[0]}}
</h3>

{% for option in q[1] %}

<label class="option">

<input type="radio"
       name="q{{q_index}}"
       value="{{loop.index0}}"
       style="width:auto">

{{["A","B","C","D"][loop.index0]}}.
{{option}}

</label>

{% endfor %}

</div>

{% endfor %}

<button type="button"
        class="btn"
        onclick="submitQuiz()">

Submit Quiz

</button>

</form>

<div id="result"></div>

</div>

</div>

<script>

let time = 300;
let submitted = false;

function timer(){

    let minutes = Math.floor(time / 60);
    let seconds = time % 60;

    document.getElementById("timer").innerHTML =
    "⏱️ Time Left: " +
    String(minutes).padStart(2,"0") +
    ":" +
    String(seconds).padStart(2,"0");

    if(time <= 0){

        submitQuiz();

    }else{

        time--;

        setTimeout(timer,1000);

    }

}

timer();

async function submitQuiz(){

    if(submitted) return;

    const resultBox = document.getElementById("result");
    const submitButton = document.querySelector("#quizForm button");

    try {
        const answers = {};
        const questionCount = {{ questions|length }};

        for(let i = 0; i < questionCount; i++){
            const selected = document.querySelector(
                'input[name="q' + i + '"]:checked'
            );
            answers[String(i)] = selected ? Number(selected.value) : null;
        }

        if(submitButton){
            submitButton.disabled = true;
            submitButton.innerText = "Submitting...";
        }

        const response = await fetch("/submit_quiz", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify({
                topic: {{ topic|tojson }},
                answers: answers
            })
        });

        const raw = await response.text();
        let data;

        try {
            data = JSON.parse(raw);
        } catch(parseError) {
            throw new Error("Server returned an invalid response. Please restart the Colab cell.");
        }

        if(!response.ok || data.error){
            throw new Error(data.error || "Quiz submission failed.");
        }

        submitted = true;

        resultBox.innerHTML = `
            <div class="result">
                <h2>🎉 Quiz Completed!</h2>
                <div class="big-score">${data.score}/${data.total}</div>
                <h2>${data.percentage}%</h2>
                <p style="color:#cbd5e1;margin:12px 0 22px">
                    Your result has been saved successfully.
                </p>
                <a class="btn" href="/dashboard">Back to Dashboard</a>
            </div>
        `;

        window.scrollTo({top: document.body.scrollHeight, behavior: "smooth"});

    } catch(error) {
        submitted = false;

        if(submitButton){
            submitButton.disabled = false;
            submitButton.innerText = "Submit Quiz";
        }

        resultBox.innerHTML = `
            <div class="card" style="margin-top:20px;border:1px solid #f87171">
                <h3 style="color:#fca5a5">⚠️ Quiz Error</h3>
                <p style="color:#fecaca;margin-top:10px">
                    ${error.message}
                </p>
                <p style="color:#cbd5e1;margin-top:10px">
                    Check that the Flask/Colab cell is still running, then try again.
                </p>
            </div>
        `;
    }
}

</script>

</body>
</html>
"""

# ============================================================
# 12. PREDICTION PAGE
# ============================================================

PREDICTION = """
<!DOCTYPE html>
<html>
<head>
<title>AI Score Prediction</title>
<style>{{css}}</style>
</head>

<body>

<div class="container">

<div class="navbar">

<div class="logo">
🤖 AI <span>PREDICTION</span>
</div>

<a class="btn" href="/dashboard">
Dashboard
</a>

</div>

<div class="card">

<h1>📊 AI Score Prediction</h1>

<p style="color:#cbd5e1;margin:15px 0">
Enter your learning details. The Machine Learning model will
estimate your expected score.
</p>

<form method="POST">

<label>Study Hours</label>
<input type="number"
       step="0.1"
       name="study_hours"
       value="5"
       min="0"
       max="20"
       required>

<label>Previous Score</label>
<input type="number"
       name="previous_score"
       value="70"
       min="0"
       max="100"
       required>

<label>Quiz Attempts</label>
<input type="number"
       name="quiz_attempts"
       value="5"
       min="0"
       max="50"
       required>

<label>Completion Percentage</label>
<input type="number"
       name="completion"
       value="80"
       min="0"
       max="100"
       required>

<label>Accuracy Percentage</label>
<input type="number"
       name="accuracy"
       value="75"
       min="0"
       max="100"
       required>

<button class="btn">
Predict Score 🚀
</button>

</form>

{% if prediction is not none %}

<div class="result">

<h2>Predicted Score</h2>

<div class="big-score">
{{prediction}}%
</div>

<p>
This is an ML-based estimate using your entered learning information.
</p>

</div>

{% endif %}

</div>

</div>

</body>
</html>
"""

# ============================================================
# 13. ROUTES
# ============================================================

@app.route("/")
def home():

    if "user_id" in session:
        return redirect("/dashboard")

    return redirect("/login")


# ---------------- LOGIN ----------------

@app.route("/login", methods=["GET","POST"])
def login():

    error = ""

    if request.method == "POST":

        email = request.form["email"].strip().lower()
        password = request.form["password"]

        conn = sqlite3.connect(DB)
        cur = conn.cursor()

        cur.execute(
            "SELECT id,name,password FROM users WHERE email=?",
            (email,)
        )

        user = cur.fetchone()

        conn.close()

        if user and check_password_hash(user[2],password):

            session["user_id"] = user[0]
            session["name"] = user[1]

            return redirect("/dashboard")

        error = "Invalid email or password."

    return render_template_string(
        LOGIN,
        css=CSS,
        error=error
    )


# ---------------- SIGNUP ----------------

@app.route("/signup", methods=["GET","POST"])
def signup():

    error = ""

    if request.method == "POST":

        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        if len(password) < 4:

            error = "Password must contain at least 4 characters."

        else:

            try:

                conn = sqlite3.connect(DB)
                cur = conn.cursor()

                cur.execute(
                    """
                    INSERT INTO users(name,email,password)
                    VALUES(?,?,?)
                    """,
                    (
                        name,
                        email,
                        generate_password_hash(password)
                    )
                )

                conn.commit()
                conn.close()

                return redirect("/login")

            except sqlite3.IntegrityError:

                error = "Email already registered."

    return render_template_string(
        SIGNUP,
        css=CSS,
        error=error
    )


# ---------------- LOGOUT ----------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")


# ---------------- DASHBOARD ----------------

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT COUNT(*), AVG(percentage), MAX(percentage)
        FROM results
        WHERE user_id=?
        """,
        (user_id,)
    )

    stats = cur.fetchone()

    conn.close()

    total_quizzes = stats[0] or 0
    avg_score = round(stats[1] or 0,1)
    best_score = round(stats[2] or 0,1)

    topic_names = list(topics.keys())

    descriptions = {
        x:topics[x]["description"]
        for x in topic_names
    }

    icons = ["🧠","🤖","📊","🌐","🔐","🐍","🗄️","💻","☁️","🧩","🌐"]

    return render_template_string(
        DASHBOARD,
        css=CSS,
        name=session["name"],
        topics=topic_names,
        descriptions=descriptions,
        icons=icons,
        total_quizzes=total_quizzes,
        avg_score=avg_score,
        best_score=best_score
    )


# ---------------- STUDY MATERIAL ----------------

@app.route("/study/<path:topic>")
def study(topic):

    if "user_id" not in session:
        return redirect("/login")

    if topic not in topics:
        return redirect("/dashboard")

    return render_template_string(
        STUDY,
        css=CSS,
        topic=topic,
        material=topics[topic]["material"]
    )


# ---------------- QUIZ ----------------

@app.route("/quiz/<path:topic>")
def quiz(topic):

    if "user_id" not in session:
        return redirect("/login")

    if topic not in topics:
        return redirect("/dashboard")

    return render_template_string(
        QUIZ,
        css=CSS,
        topic=topic,
        questions=topics[topic]["questions"]
    )


# ---------------- SUBMIT QUIZ ----------------

@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():

    if "user_id" not in session:
        return jsonify({"error":"Login required. Please log in again."}), 401

    try:
        data_json = request.get_json(silent=True)

        if not isinstance(data_json, dict):
            return jsonify({"error":"Invalid quiz request."}), 400

        topic = data_json.get("topic")
        answers = data_json.get("answers", {})

        if topic not in topics:
            return jsonify({"error":"Invalid topic. Please return to the dashboard."}), 400

        if not isinstance(answers, dict):
            return jsonify({"error":"Invalid answers format."}), 400

        questions = topics[topic]["questions"]
        score = 0

        for i, q in enumerate(questions):
            correct_answer = int(q[2])
            user_answer = answers.get(str(i))

            if user_answer is None or user_answer == "":
                continue

            try:
                if int(user_answer) == correct_answer:
                    score += 1
            except (ValueError, TypeError):
                continue

        total = len(questions)
        percentage = round((score / total) * 100, 2) if total else 0

        conn = sqlite3.connect(DB)
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO results
            (user_id,topic,score,total,percentage)
            VALUES(?,?,?,?,?)
            """,
            (
                session["user_id"],
                topic,
                score,
                total,
                percentage
            )
        )

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "score": score,
            "total": total,
            "percentage": percentage
        }), 200

    except Exception as e:
        print("QUIZ SUBMISSION ERROR:", repr(e))
        return jsonify({
            "error":"Internal quiz error. Please restart the Flask/Colab cell and try again."
        }), 500


@app.errorhandler(500)
def handle_server_error(error):
    print("SERVER ERROR:", repr(error))
    if request.path == "/submit_quiz":
        return jsonify({"error":"Server error while submitting the quiz. Please restart the Colab cell."}), 500
    return "Internal server error.", 500


# ---------------- SCORE PREDICTION ----------------

@app.route("/prediction", methods=["GET","POST"])
def prediction():

    if "user_id" not in session:
        return redirect("/login")

    result = None

    if request.method == "POST":

        values = [[
            float(request.form["study_hours"]),
            float(request.form["previous_score"]),
            float(request.form["quiz_attempts"]),
            float(request.form["completion"]),
            float(request.form["accuracy"])
        ]]

        result = round(
            float(model.predict(pd.DataFrame(values, columns=[
                "study_hours","previous_score","quiz_attempts","completion","accuracy"
            ]))[0]),
            1
        )

        result = max(0,min(100,result))

    return render_template_string(
        PREDICTION,
        css=CSS,
        prediction=result
    )


# ============================================================
# 14. START FLASK
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )
