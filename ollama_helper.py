import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_plant_report(disease_name, confidence, severity_hint):

    if disease_name.lower() == "healthy":
        prompt = f"""
        A plant classification system predicted:

        Condition: Healthy
        Model confidence: {confidence:.2f}

        The plant shows no visible disease symptoms.

        Provide:

        - Short confirmation message (2-3 lines)
        - Key preventive care steps
        - Monitoring recommendations
        - When the farmer should recheck

        Keep response concise and practical.
        Do NOT provide general farming guide.
        """

    else:
        prompt = f"""
        A plant disease classification system predicted:

        Disease: {disease_name}
        Model confidence: {confidence:.2f}
        Severity hint from model: {severity_hint}

        Act as an agricultural expert.

        Respond STRICTLY in this format:

        Disease Summary:
        Causes:
        Organic Treatment:
        Chemical Treatment:
        Prevention:
        Severity Level (Low/Moderate/High):
        Estimated Crop Impact:
        Farmer Advice:

        Keep explanation practical and concise.
        """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Error connecting to Ollama."