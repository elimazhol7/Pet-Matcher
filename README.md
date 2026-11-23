# 🐾 Pet Matcher
** Find your perfect animal companion with Machine Learning **

---
## Features

- **Interactive quiz:** dropdown questions and a one-click "Find My Perfect Pet!" button, built with `ipywidgets`
- **Machine learning matching:** a scikit-learn KNN pipeline with feature scaling and one-hot encoding
- **10 possible matches:** Dog, Cat, Rabbit, Hamster, Guinea pig, Bird, Fish, Turtle, Leopard gecko and Ferret
- **Personalized results:** each match comes with a short description and a care guide
- **A responsible-ownership reminder:** every result encourages you to research the animal's needs before adopting

---


## 🔮 Future Improvements

- Show the **top 3 matches** with confidence scores (using `predict_proba`)
- Add more pets and species-specific options (e.g. different dog sizes or bird species)
- Collect real survey data instead of synthetic examples
- Explain *which* answers drove the match
- Add questions about budget, allergies and how long the pet lives
- Deploy as a web app with Streamlit or Gradio

---

## 🛠️ Built With

- [pandas](https://pandas.pydata.org/): data handling
- [scikit-learn](https://scikit-learn.org/): preprocessing and KNN model
- [ipywidgets](https://ipywidgets.readthedocs.io/): interactive user interface
