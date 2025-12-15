# API Documentation

## Alzheimer's Disease Detection Platform API

**Version**: 1.0.0  
**Base URL**: `http://localhost:8000`  
**Authentication**: Bearer Token (API Key)

---

## 📋 **Table of Contents**

1. [Authentication](#authentication)
2. [Endpoints](#endpoints)
3. [Data Models](#data-models)
4. [Error Handling](#error-handling)
5. [Rate Limiting](#rate-limiting)
6. [Examples](#examples)

---

## 🔐 **Authentication**

All API endpoints require authentication using a Bearer token.

### Getting API Token

```bash
# Request token (in production, this would be through a secure portal)
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "clinic_user", "password": "secure_password"}'
```

### Using API Token

```bash
# Include token in Authorization header
curl -X GET http://localhost:8000/model/info \
  -H "Authorization: Bearer your-api-token-here"
```

---

## 🎯 **Endpoints**

### 1. Health Check

Check API health and system status.

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "device": "cuda:0",
  "timestamp": "2025-01-15T10:30:00"
}
```

---

### 2. Get Model Information

Retrieve model version, performance metrics, and capabilities.

```http
GET /model/info
```

**Response**:
```json
{
  "version": "1.0.0",
  "architecture": "Multimodal CNN + Transformer + Attention Fusion",
  "training_date": "2025-01-15",
  "performance_metrics": {
    "accuracy": 0.94,
    "sensitivity": 0.92,
    "specificity": 0.95,
    "auc_roc": 0.97,
    "precision": 0.91,
    "recall": 0.92,
    "f1_score": 0.91
  },
  "supported_modalities": ["MRI", "PET", "Clinical", "Cognitive"],
  "target_accuracy": "Research goal: 99% under controlled experimental conditions",
  "medical_disclaimer": "This AI system provides clinical decision support only. All predictions must be reviewed by qualified healthcare professionals."
}
```

---

### 3. Single Patient Prediction

Main endpoint for Alzheimer's disease prediction.

```http
POST /predict
```

**Request Body** (multipart/form-data):
```
mri_file: <binary MRI NIfTI file>
pet_file: <binary PET NIfTI file>
clinical_data: <JSON string with clinical assessments>
patient_id: <string patient identifier>
```

**Clinical Data JSON Schema**:
```json
{
  "age": 75.0,
  "gender": 0,
  "education": 16.0,
  "mmse": 24.0,
  "cdr": 0.5,
  "adas_cog": 18.0,
  "faq": 8.0,
  "mem_delay": 15.0,
  "mem_imm": 20.0,
  "boston": 25.0,
  "trail_a": 45.0,
  "trail_b": 95.0,
  "digit_forward": 8.0,
  "digit_backward": 6.0,
  "category_fluency": 18.0,
  "phonemic_fluency": 22.0,
  "gds": 2.0,
  "naccdd": 1.0,
  "cvd": 0.0,
  "diabetes": 0.0
}
```

**Response**:
```json
{
  "patient_id": "PAT-001",
  "prediction": "Mild Cognitive Impairment",
  "probabilities": {
    "Cognitively_Normal": 0.15,
    "Mild_Cognitive_Impairment": 0.72,
    "Alzheimers_Disease": 0.13
  },
  "confidence": 0.72,
  "uncertainty": 0.08,
  "risk_score": 0.85,
  "recommendation": "Consider cognitive training, lifestyle interventions, and regular follow-up.",
  "timestamp": "2025-01-15T10:30:00",
  "model_version": "1.0.0",
  "processing_time": 2.3
}
```

**Response Codes**:
- `200`: Successful prediction
- `400`: Invalid input data
- `401`: Authentication required
- `500`: Internal server error

---

### 4. Batch Prediction

Process multiple patients in batch (research use).

```http
POST /batch_predict
```

**Request Body** (multipart/form-data):
```
files: <array of MRI/PET file pairs>
clinical_data: <array of clinical data JSON strings>
research_study_id: <string research identifier>
```

**Response**:
```json
{
  "batch_id": "BATCH-001",
  "total_processed": 50,
  "successful": 48,
  "failed": 2,
  "results": [
    {
      "patient_id": "PAT-001",
      "prediction": "Cognitively Normal",
      "confidence": 0.89,
      "uncertainty": 0.05
    }
  ],
  "processing_time": 120.5,
  "status": "completed"
}
```

---

### 5. Get Model Explanation

Retrieve model explanation for a specific prediction.

```http
GET /explain/{patient_id}
```

**Response**:
```json
{
  "patient_id": "PAT-001",
  "explanation": {
    "attention_maps": {
      "mri": "<base64 encoded attention map>",
      "pet": "<base64 encoded attention map>"
    },
    "feature_importance": {
      "hippocampus_volume": 0.23,
      "mmse_score": 0.18,
      "cdr_score": 0.15,
      "pet_temporal": 0.12,
      "education_years": 0.08
    },
    "clinical_reasoning": "Prediction is based on moderate hippocampal atrophy and mild cognitive decline on MMSE.",
    "uncertainty_breakdown": {
      "data_quality": 0.03,
      "model_confidence": 0.05
    }
  },
  "medical_disclaimer": "Explanations are for research and educational purposes. Clinical interpretation required."
}
```

---

### 6. Submit Feedback

Submit clinical feedback for model improvement.

```http
POST /feedback
```

**Request Body**:
```json
{
  "patient_id": "PAT-001",
  "feedback": "Model correctly identified MCI, confirmed by neurologist",
  "correct_diagnosis": "Mild Cognitive Impairment",
  "confidence_rating": 4,
  "clinical_notes": "Patient shows mild memory complaints but maintains daily function",
  "user_id": "dr_smith"
}
```

**Response**:
```json
{
  "status": "feedback_received",
  "message": "Thank you for your feedback. This helps improve the system.",
  "next_steps": "Feedback will be reviewed by clinical experts and used for model improvement.",
  "feedback_id": "FB-001"
}
```

---

### 7. System Metrics

Get real-time system performance metrics.

```http
GET /metrics
```

**Response**:
```json
{
  "system_status": "operational",
  "predictions_today": 42,
  "predictions_this_week": 312,
  "predictions_this_month": 1247,
  "average_confidence": 0.89,
  "average_processing_time": 2.3,
  "model_uptime": "99.9%",
  "clinical_feedback": {
    "total": 89,
    "positive": 85,
    "negative": 4
  },
  "last_updated": "2025-01-15T10:30:00"
}
```

---

## 📊 **Data Models**

### Prediction Classes
- **Cognitively Normal (CN)**: No cognitive impairment
- **Mild Cognitive Impairment (MCI)**: Subtle cognitive changes, not dementia
- **Alzheimer's Disease (AD)**: Meets criteria for Alzheimer's dementia

### Confidence Levels
- **High (≥0.8)**: Model is confident in prediction
- **Medium (0.5-0.8)**: Moderate confidence, review recommended
- **Low (<0.5)**: Low confidence, human review required

### Risk Score Interpretation
- **0.0-0.3**: Low risk (CN likely)
- **0.3-0.7**: Moderate risk (MCI possible)
- **0.7-1.0**: High risk (AD likely)

---

## ❌ **Error Handling**

### Error Response Format
```json
{
  "error": "Invalid input data",
  "message": "MMSE score must be between 0 and 30",
  "status_code": 400,
  "request_id": "req-12345"
}
```

### Common Error Codes
- `400`: Bad Request - Invalid input data
- `401`: Unauthorized - Invalid or missing token
- `403`: Forbidden - Insufficient permissions
- `404`: Not Found - Resource not found
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Server error
- `503`: Service Unavailable - System maintenance

---

## ⏱️ **Rate Limiting**

### Limits
- **Standard users**: 100 requests/hour
- **Premium users**: 500 requests/hour
- **Research users**: 1000 requests/hour
- **Batch processing**: 10 requests/minute

### Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1642680000
```

---

## 💡 **Examples**

### Python Client Example

```python
import requests
import json

class AlzheimerAPIClient:
    def __init__(self, base_url, api_token):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_token}'
        }
    
    def predict(self, mri_path, pet_path, clinical_data, patient_id):
        """Make prediction for single patient"""
        
        with open(mri_path, 'rb') as mri_file, open(pet_path, 'rb') as pet_file:
            files = {
                'mri_file': mri_file,
                'pet_file': pet_file
            }
            
            data = {
                'clinical_data': json.dumps(clinical_data),
                'patient_id': patient_id
            }
            
            response = requests.post(
                f'{self.base_url}/predict',
                files=files,
                data=data,
                headers=self.headers
            )
            
            response.raise_for_status()
            return response.json()
    
    def get_explanation(self, patient_id):
        """Get model explanation"""
        response = requests.get(
            f'{self.base_url}/explain/{patient_id}',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage
client = AlzheimerAPIClient('http://localhost:8000', 'your-api-token')

# Clinical data
clinical_data = {
    'age': 75.0,
    'gender': 0,
    'education': 16.0,
    'mmse': 24.0,
    'cdr': 0.5,
    # ... other fields
}

# Make prediction
result = client.predict(
    mri_path='patient_mri.nii.gz',
    pet_path='patient_pet.nii.gz',
    clinical_data=clinical_data,
    patient_id='PAT-001'
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### JavaScript Client Example

```javascript
class AlzheimerAPIClient {
    constructor(baseUrl, apiToken) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${apiToken}`
        };
    }
    
    async predict(mriFile, petFile, clinicalData, patientId) {
        const formData = new FormData();
        formData.append('mri_file', mriFile);
        formData.append('pet_file', petFile);
        formData.append('clinical_data', JSON.stringify(clinicalData));
        formData.append('patient_id', patientId);
        
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            headers: this.headers,
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async getExplanation(patientId) {
        const response = await fetch(`${this.baseUrl}/explain/${patientId}`, {
            headers: this.headers
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
}

// Usage
const client = new AlzheimerAPIClient('http://localhost:8000', 'your-api-token');

// Make prediction
const result = await client.predict(
    mriFile,
    petFile,
    clinicalData,
    'PAT-001'
);

console.log(`Prediction: ${result.prediction}`);
console.log(`Confidence: ${result.confidence.toFixed(3)}`);
```

---

## 🔧 **SDKs and Libraries**

### Official SDKs
- **Python**: `pip install alzheimer-api-client`
- **JavaScript**: `npm install alzheimer-api-client`
- **R**: `install.packages("alzheimer.api")`

### Community SDKs
- **Java**: Community-maintained
- **C#**: Community-maintained
- **Go**: Community-maintained

---

## 📞 **Support**

### API Support
- **Documentation**: docs.alzheimer-ai-platform.org/api
- **Issues**: GitHub Issues
- **Email**: api-support@alzheimer-ai-platform.org

### Clinical Support
- **Training**: training@alzheimer-ai-platform.org
- **Safety**: safety@alzheimer-ai-platform.org

---

## 📄 **Changelog**

### Version 1.0.0 (2025-01-15)
- Initial API release
- Support for MRI, PET, and clinical data
- Real-time prediction endpoint
- Model explanation capabilities
- Clinical feedback system

### Planned Features
- Longitudinal tracking
- Additional imaging modalities
- Multi-language support
- Mobile SDK

---

**Last Updated**: 2025-01-15  
**Documentation Version**: 1.0.0