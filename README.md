Agent for DotNet project, but can be changed for other languages, just need to change the Dockerfile & Code Path.

Create virtual environment
Install dependencies (Not all present in requirements.txt)
export GEMINI_API_KEY=""
Change this line in docker-compose.yml (- /Users/saurabhgupta/Saurabh/L&T Finance/Coding/SamplePaymentService/DotNet/PaymentService:/workspace)
python -m agent.main 