# iQvault.ai 🧠

**Your documents. Now intelligent.**

A secure AI assistant that reads your **PDFs, Word files, PowerPoint slides, and text files** — and answers questions like a human.
  
No cloud.  
No data leaks.  
Just answers — from your own documents.

Perfect for:
- Presales teams
- Consultants
- HR departments
- Students
- Anyone who works with documents

---

## 🚀 How to Run (Step-by-Step)

You don’t need to be a coder. Just follow these steps:

### 1. Download the App
- Click **"Code" → "Download ZIP"** on the GitHub page
- Extract the folder to your computer (e.g., `D:\Documents\iQvault.ai`)

### 2. Download the AI Model (One-Time)
The AI model is too big to include, so you download it separately.

1. Go to:  
   👉 [https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
2. Click **`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`** to download
 Save it to:  model (for Users)
 This is the only file you need to download manually.

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt

 4. Run the App
Double-click the file:
👉 run.bat

It will:

Open a command window
Start the app
Launch your browser
Go to: http://localhost:8501

5. Add Your Documents
Put your files in the data/ folder:

report.pdf
proposal.docx
presentation.pptx
notes.txt
You can add as many as you want.

No need to convert or rename them.

6.  Ask Questions
Now ask anything, like:

“What was in the Q3 strategy?”
“How did we reduce cloud costs?”
“Summarize the client proposal”
The AI will:

Search your documents
Generate an answer
Show the exact source (so you can verify it)

7. ⚠️ Troubleshooting (Common Fixes)
❌ "pip is not recognized"
You don’t have Python installed properly
Download Python from: https://python.org
Make sure to check "Add Python to PATH" during installation
❌ "Model path doesn't exist"
The .gguf file is not in the model/ folder
Double-check the path:
iQvault.ai\model\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
❌ App crashes or won’t start
Close all Word, PowerPoint, or PDF files that might be locked
Restart and try again

8.  Want to Help Improve iQvault.ai?
This is open source — you can:

Report bugs
Suggest features (e.g., Excel support)
Help make a Mac/Linux version
Translate it into other languages
Just open an issue or pull request on GitHub.

9.🌐 Connect
Built by: Arjun Lande
GitHub: github.com/arjunlande21/iQvault.ai
Have questions? Want to collaborate?
Feel free to reach out!

 10. Usage Analytics
This app includes **anonymous usage tracking** to help improve it.
- Only OS and version are sent
- No documents, questions, or personal data
- You can disable it by editing `app.py`

License

MIT — use, modify, and share freely.


