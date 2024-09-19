import pdfplumber
from flask import Flask, render_template, request, jsonify
from main import get_ai_search_instance

app = Flask(__name__)
ai_search = get_ai_search_instance()


@app.route('/')
def home():
    return render_template('index.html')


def extract_text_from_pdf(file):
    # Extract text from PDF file using pdfplumber
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files.get('file')
    if not file:
        return jsonify({'success': False, 'message': 'No file provided.'}), 400

    try:
        pdf_text = extract_text_from_pdf(file)
        ai_search.store_new_embedding(pdf_text, file.filename)

        return jsonify({'success': True, 'message': 'File uploaded and ingested successfully!'})

    except Exception as e:
        # Handle errors during the file processing
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    query = request.json.get('query', '')
    if not query:
        return jsonify({'response': 'No query provided.'}), 400

    context, source = ai_search.answer_retriever(query)

    final_answer = "It seems there's no information related to your query at the moment. " \
                   "If you have specific documents, please upload them to enrich my knowledge base."
    if context:
        final_answer = ai_search.augmentation(context, query)

    return jsonify({'response': final_answer})


if __name__ == '__main__':
    app.run(debug=True)
