# OkCupid Stylometry Analysis Project

Version 0.1.0
# OkCupid classifier

This project aims to examine to what extent educational background can be inferred from the written text, based on the assumption that educational levels are associated with the style of writing. This includes people's signature fashion of using certain vocabulary of words which makes their literature unique and recognizable. Using a large public dataset of almost 60000 dating profiles, we aimed to model author style to be used as a predictor of educational background.
In this project we have used a text analysis program named LIWC(Linguistic Inquiry and Word Count, https://liwc.wpengine.com/) to extract the degree of various categories of words that are used in the userwritten text. With adding these linguistic features to text features we have trained a logistic regression model. The model predicts the user's educational background with accuracy of 83%.

## Researcher & engineers

Researcher:
- Dr. Corten, R. (Rense)

Project Manager:
- Dr. Laurence Frank

Data Engineer:
- Shiva Nadi


## Installation

This project requires Python 3.7.3 or higher. Install the dependencies with the
code below.

```sh
pip install -r requirements.txt
```

## License

This project is licensed under the terms of the [MIT License](/LICENSE.md)

## Citation

Please [cite this project as described here](/CITATION.md).
