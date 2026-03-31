#  Autonomous Cloud Cost Intelligence Platform

> An autonomous system that monitors cloud infrastructure metrics, detects cost anomalies using machine learning, and lays the foundation for self-healing cloud environments without human intervention.

--- 
 <h5> Before we proceed, please do read the final keynote at the end of this document. It's an invaluable doctrine I'd like to share. </h5>

##  Problem Statement

Cloud infrastructure costs are notoriously difficult to monitor at scale. Unexpected spikes in CPU usage, runaway serverless functions, idle instances, and traffic surges can silently drain cloud budgets before any human notices. Existing solutions require manual threshold configuration and reactive human intervention.

**The Software** addresses this by building an autonomous cost intelligence pipeline that continuously watches cloud metrics, learns what "normal" looks like, flags genuine anomalies using ML, and is designed to automatically trigger corrective actions all in real time.

---

##  What It Does

```
Collect Metrics → Detect Anomalies → Decide Action → Show Impact
```

1. **Simulate / Ingest** cloud telemetry data (CPU, memory, network, cost, requests)
2. **Detect** anomalies using a Z-score based Logistic Regression model
3. **Decide** what corrective action to take (stop idle instance, limit function calls, clean storage)
4. **Visualise** cost trends, anomaly scores, and before/after impact on a real-time dashboard

---

## 🧠 ML Approach & Design Decisions

### Why Simulated Data?
Integrating a live AWS/GCP billing API introduces significant complexity authentication, SDK configuration, rate limits, and billing delays. For this proof of concept, we simulate realistic cloud telemetry using controlled statistical distributions. The architecture is identical to a live system; only the data source changes.

### Why Z-Score?
Z-score is a statistically grounded method to identify outliers data points that deviate significantly from the mean. Plotting Z-scores derived from our simulated dataset clearly reveals cost spikes, providing clean ground-truth labels to train our ML model.

### Why Linear Regression First?
We initially tested Linear Regression as a baseline classifier. While functional, it is fundamentally a regression tool not designed for binary classification of rare events like anomalies. It was retained as a documented baseline.

### Why Logistic Regression (Secondary Model)?
Logistic Regression is natively designed for binary classification. It outputs a probability between 0 and 1, handles the normal/anomaly classification task correctly, and critically its coefficients are fully interpretable. We can show exactly which cloud metrics (CPU, memory, network) contribute most to predicting a cost anomaly, giving actionable insight rather than a black-box alert like the Isolation forest would.

### Why Isolation Forest? (Primary anomaly detection model):
Primarily because the isolation forest is capable of pointing out anomalies of all nature for a particular metric, even ones that may not have appeared in the past. If, say for example, a certain kind of drop in CPU levels goes undetected by LR, the isolation forest would still catch it, even if it's a new kind of idle drop that hasn't happened before, this is the biggest advantage of integrating the isolation forest model alongside a classifcation type model. 

---

##  Future Integrations (Roadmap)


### Phase 2 — Live Cloud API Integration (Now complete)
- Connect to real AWS CloudWatch 
- Replace simulated data with live telemetry streams
- Retrain the model on real usage patterns

### Phase 3 — Autonomous Remediation
- When an anomaly is confirmed, automatically trigger corrective backend actions via cloud APIs:
  - Stop idle EC2/GCP instances (Complete)
  - POTENTIAL FUTURE INTEGRATIONS:
  - Limit Lambda/Cloud Function concurrency
  - Clean unused S3/GCS storage volumes
  - Tag unreviewed assets for audit
- Full closed loop: **Detect → Decide → Act** with no human intervention 

### Phase 4: Replacing Logistic regression with XgBoost: 
The boosted gradient trees, or XgBoost is today's meta amongst forest based decision making classifiers, primarily due to its ability to grow increasingly accurate with time, correcting itself with every new data point. It can handle more complex relationships when server metrics are interconnected or related, as seen in a real life scenario. This model would truly push our product to near production-grade, however, getting it to work on the AWS free tier EC2 instances is near imposssible, we need more computing power, in fact, even the installation of the XgBoost package fails because of the limited memory this free server instance provides.

---

👥 Team & Roles
| Member | Role |
|---|---|
| Suryaansh and Rayaan| ML Engineering — model training, Z-score labelling, Logistic Regression |
| Shanmukesh | Cloud Integration — AWS/GCP API research, SDK setup, live data pipeline |
| Rohan | Cloud Integration — remediation engine design, backend action automation |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Simulation | Python, NumPy, Pandas |
| Anomaly Detection | Scikit-learn (Logistic Regression, StandardScaler) |
| Statistical Analysis | SciPy (Z-score) |
| Dashboard | Streamlit, Matplotlib |
| Version Control | Git, GitHub|
|Virtual Server | AWS-EC2|

---



##  Requirements

```
numpy
pandas
scipy
scikit-learn
matplotlib
streamlit
```

--- 

## Access Dashboard: 
http://3.19.32.216:8502/ 

---

### The Final Keynote (Raw opinion): 
Why'd we only place 13th? What did I do wrong that the others did right, even with their product being as simple as our first prototype? 

The answer to our loss lies in the final round's presentation. 
I presented well, I'm sure of it, but I channelled all that speech skill in the wrong direction. I came across this revelation much later, after enduring the loss. 

<strong>HACKATHONS</strong> don't give a damn about the technical complexity of your project, built a 100B parameter, multi-modal LLM from scratch but it can't save a company's money? <em>Disqualifed.</em> Built an isolation forest that can reduces costs from $x to $x-10000? You're on the pedestal, that's first place. 

This is how it has always worked and it will continue to function this way forever. 
How would your system, irrespective of its technical proficiency on paper, perform in real life when exposed to real data under unstable circumstances? And more importantly, does it add to the bills or does it save them a fortune? These are the only ways hackathons distinguish between what's good and what's best. 

So no matter what you build, speak like a salesperson whilst you're presenting, show the evaluators how your creations can make them rich, sell it to them like you're selling them a dream and you'd win absolutely any hackathon on the surface of this planet. It's harsh but it's real. 

Thank you.