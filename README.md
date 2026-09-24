# Spam Prediction Dashboard — Unit 5 Project

A minimal end-to-end project matching **"Displaying ML/AI Outputs on Web"**:
Fetching data from APIs → Rendering model predictions → Interactive visualization with Chart.js.

## What's inside
- **`app.py`** — Flask backend. Trains a tiny Naive Bayes spam-classifier at startup and serves it as a REST API.
  - `GET /api/predict?text=your+message` → `{ "prediction": "Spam", "confidence": 0.92 }`
  - `GET /api/predictions` → batch predictions for a demo "inbox" (used by the dashboard)
- **`index.html`** — Frontend dashboard. Fetches from the API and renders:
  1. A one-line text summary (`textContent`)
  2. A dynamic table with a "Remove" button per row
  3. A Chart.js bar chart, color-coded red/green by prediction, with a "Refresh" button that re-fetches and redraws

This is exactly the pattern from section 5.7 of your notes — the only difference is the API is now **real**, not a placeholder.

## How to run it

1. Install dependencies:
   ```bash
   pip install flask flask-cors scikit-learn
   ```
2. Start the backend:
   ```bash
   python app.py
   ```
   This runs the API at `http://127.0.0.1:5000`.
3. Open `index.html` in your browser (just double-click it, or use a simple local server).
4. Click **Refresh Predictions** to re-fetch and see the table/chart update.

## How this maps to the chapter

| Concept in Unit 5 | Where it is in this project |
|---|---|
| 5.1 Fetch API for predictions | `index.html` → `fetch(API_URL)` |
| 5.2 Rendering with DOM (`textContent`) | `index.html` → `#summary` text |
| 5.2.2 Dynamic table (insertRow/insertCell) | `index.html` → table-building code |
| 5.4 Chart.js bar chart | `index.html` → `new Chart(...)` |
| 5.4.5 Updating a chart dynamically | `Refresh` button → `resultsChart.destroy()` + redraw |
| "Model deployed behind an API" | `app.py` → trained `MultinomialNB` model served via Flask |

## Extending it (matches the Practice Exercises)
- Swap `app.py`'s toy dataset for a real dataset (e.g. SMS Spam Collection) for a more accurate model.
- Add a Plotly.js version of the chart (section 5.5) alongside or instead of Chart.js, and compare code length/interactivity.
- Point `index.html` at `https://jsonplaceholder.typicode.com/users` instead, to do Exercise 1 (table of name/city with delete buttons).
