import streamlit as st 

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Transformers
from transformers import pipeline

import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")

# Web Scraping Pkg
from bs4 import BeautifulSoup
from urllib.request import urlopen

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""



# Sumy
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


# Transformer (cached)
@st.cache_resource
def load_summarizer():
	return pipeline("summarization")

def transformer_summarizer(text):
	summarizer = load_summarizer()

	text = text[:2000]

	result = summarizer(text, max_length=130, min_length=30, do_sample=False)
	return result[0]['summary_text']


@st.cache_data
def get_text(raw_url):
	page = urlopen(raw_url)
	soup = BeautifulSoup(page, "html.parser")
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text


@st.cache_resource
def analyze_text(text):
	return nlp(text)




def main():
	"""Summarizer Streamlit App"""

	st.title("Summarizer and Entity Checker")

	activities = ["Summarize","NER Checker","NER For URL"]
	choice = st.sidebar.selectbox("Select Activity",activities)

	
	if choice == 'Summarize':
		st.subheader("Summarize Document")
		raw_text = st.text_area("Enter Text Here","Type Here")

		summarizer_type = st.selectbox(
			"Summarizer Type",
			["Sumy Lex Rank", "Transformer (BART)"]
		)

		if st.button("Summarize"):

			if raw_text.strip() == "":
				st.warning("Please enter some text")

			else:
				if summarizer_type == "Sumy Lex Rank":
					summary_result = sumy_summarizer(raw_text)

				elif summarizer_type == "Transformer (BART)":
					with st.spinner("Running transformer model..."):
						summary_result = transformer_summarizer(raw_text)

				st.write(summary_result)


	
	if choice == 'NER Checker':
		st.subheader("Named Entity Recog with Spacy")
		raw_text = st.text_area("Enter Text Here","Type Here")

		if st.button("Analyze"):
			docx = analyze_text(raw_text)
			html = displacy.render(docx,style="ent")
			html = html.replace("\n\n","\n")
			st.write(HTML_WRAPPER.format(html),unsafe_allow_html=True)


	
	if choice == 'NER For URL':
		st.subheader("Analysis on Text From URL")
		raw_url = st.text_input("Enter URL Here","Type here")
		text_preview_length = st.slider("Length to Preview",50,100)

		if st.button("Analyze"):
			if raw_url != "Type here":
				result = get_text(raw_url)

				len_of_full_text = len(result)
				len_of_short_text = round(len(result)/text_preview_length)

				st.success(f"Length of Full Text: {len_of_full_text}")
				st.success(f"Length of Short Text: {len_of_short_text}")

				st.info(result[:len_of_short_text])

				summarized_docx = sumy_summarizer(result)

				docx = analyze_text(summarized_docx)
				html = displacy.render(docx,style="ent")
				html = html.replace("\n\n","\n")

				st.write(HTML_WRAPPER.format(html),unsafe_allow_html=True)


if __name__ == '__main__':
	main()