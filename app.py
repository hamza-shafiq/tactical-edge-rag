import pdfplumber
from flask import Flask, render_template, request, jsonify
from main import get_ai_search_instance

app = Flask(__name__)
ai_search = get_ai_search_instance()


@app.route('/')
def home():
    """
    Render the home page with an index template.
    """
    return render_template('index.html')


def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file.
    """
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


@app.route('/upload', methods=['POST'])
def upload_pdf():
    """
    Endpoint for uploading a PDF file, extracting its content &
    storing it as an embedding in the AI search instance.
    """
    file = request.files.get('file')
    if not file:
        return jsonify({'success': False, 'message': 'No file provided.'}), 400

    try:
        pdf_text = extract_text_from_pdf(file)
        ai_search.store_new_embedding(pdf_text, file.filename)
        ai_search.initialize_retriever()

        return jsonify({'success': True, 'message': 'File uploaded and ingested successfully!'})

    except Exception as ex:
        return jsonify({'success': False, 'message': str(ex)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint for handling chat queries. It uses the AI search instance to find
    context based on the provided query and generates a response.
    """
    query = request.json.get('query', '')
    if not query:
        return jsonify({'response': 'No query provided.'}), 400

    try:
        context, source = ai_search.answer_retriever(query)

        final_answer = "It seems there's no information related to your query at the moment. " \
                       "If you have specific documents, please upload them to enrich my knowledge base."
        if context:
            final_answer = ai_search.augmentation(context, query)

        return jsonify({'response': final_answer})

    except Exception as ex:
        return jsonify({'response': f"Exception raised! Error: {str(ex)}"})


if __name__ == '__main__':
    app.run(debug=True)
