/**
 * Clinical Dashboard JavaScript
 * Handles UI interactions, API calls, and data visualization
 */

// API configuration
const API_BASE_URL = 'http://localhost:8000';
const API_TOKEN = 'demo-clinic-token-123'; // In production, this would be securely stored

// Global variables
let currentPatientId = null;
let currentPrediction = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    loadRecentPredictions();
    loadPerformanceMetrics();
});

// Initialize dashboard components
function initializeDashboard() {
    console.log('Initializing Clinical Dashboard...');
    
    // Check API connectivity
    checkAPIHealth();
    
    // Load initial data
    updateQuickStats();
    
    console.log('Dashboard initialized successfully');
}

// Setup event listeners
function setupEventListeners() {
    // Form submission
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', handlePredictionSubmit);
    
    // File upload handlers
    const mriFile = document.getElementById('mriFile');
    const petFile = document.getElementById('petFile');
    
    mriFile.addEventListener('change', function() {
        const fileName = this.files[0]?.name || '';
        document.getElementById('mriFileName').textContent = fileName ? `Selected: ${fileName}` : '';
    });
    
    petFile.addEventListener('change', function() {
        const fileName = this.files[0]?.name || '';
        document.getElementById('petFileName').textContent = fileName ? `Selected: ${fileName}` : '';
    });
    
    // Clear form button
    const clearButton = document.querySelector('button[type="button"]');
    clearButton.addEventListener('click', clearForm);
    
    // Action buttons in results panel
    document.addEventListener('click', function(e) {
        if (e.target.textContent.includes('Generate Report')) {
            generateReport();
        } else if (e.target.textContent.includes('View Explanation')) {
            viewExplanation();
        } else if (e.target.textContent.includes('Add Feedback')) {
            addFeedback();
        }
    });
}

// Check API health
async function checkAPIHealth() {
    try {
        const response = await axios.get(`${API_BASE_URL}/health`, {
            headers: { 'Authorization': `Bearer ${API_TOKEN}` }
        });
        
        if (response.data.status === 'healthy') {
            console.log('API connection healthy');
        } else {
            console.warn('API health check warning:', response.data);
        }
    } catch (error) {
        console.error('API health check failed:', error);
        showNotification('Warning: Could not connect to prediction service', 'warning');
    }
}

// Update quick stats
async function updateQuickStats() {
    try {
        const response = await axios.get(`${API_BASE_URL}/metrics`, {
            headers: { 'Authorization': `Bearer ${API_TOKEN}` }
        });
        
        const data = response.data;
        
        // Update stats with animation
        animateNumber('todayPredictions', data.predictions_today || 12);
        animateNumber('highConfidence', Math.round((data.average_confidence || 0.89) * 100) + '%');
        animateNumber('reviewRequired', data.review_required || 3);
        animateNumber('avgProcessing', (data.average_processing_time || 2.3) + 's');
        
    } catch (error) {
        console.error('Failed to update stats:', error);
    }
}

// Animate number counting
function animateNumber(elementId, targetValue) {
    const element = document.getElementById(elementId);
    const isPercentage = typeof targetValue === 'string' && targetValue.includes('%');
    const isDecimal = typeof targetValue === 'string' && targetValue.includes('.');
    const numericValue = parseFloat(targetValue.toString().replace(/[^\d.]/g, ''));
    
    let current = 0;
    const increment = numericValue / 30;
    const timer = setInterval(() => {
        current += increment;
        if (current >= numericValue) {
            current = numericValue;
            clearInterval(timer);
        }
        
        let displayValue = Math.floor(current);
        if (isDecimal) displayValue = current.toFixed(1);
        if (isPercentage) displayValue += '%';
        
        element.textContent = displayValue;
    }, 50);
}

// Handle form submission
async function handlePredictionSubmit(e) {
    e.preventDefault();
    
    // Validate form
    if (!validateForm()) {
        return;
    }
    
    // Show loading state
    showLoadingState();
    
    try {
        // Prepare form data
        const formData = new FormData();
        
        // Add files
        const mriFile = document.getElementById('mriFile').files[0];
        const petFile = document.getElementById('petFile').files[0];
        formData.append('mri_file', mriFile);
        formData.append('pet_file', petFile);
        
        // Add clinical data
        const clinicalData = {
            age: parseFloat(document.getElementById('age').value),
            gender: parseInt(document.getElementById('gender').value),
            education: parseFloat(document.getElementById('education').value),
            mmse: parseFloat(document.getElementById('mmse').value),
            cdr: parseFloat(document.getElementById('cdr').value),
            adas_cog: parseFloat(document.getElementById('adasCog').value),
            faq: parseFloat(document.getElementById('faq').value),
            mem_delay: 15.0, // Default values for demo
            mem_imm: 20.0,
            boston: 25.0,
            trail_a: 45.0,
            trail_b: 95.0,
            digit_forward: 8.0,
            digit_backward: 6.0,
            category_fluency: 18.0,
            phonemic_fluency: 22.0,
            gds: 2.0,
            naccdd: 1.0,
            cvd: 0.0,
            diabetes: 0.0
        };
        
        formData.append('clinical_data', JSON.stringify(clinicalData));
        formData.append('patient_id', document.getElementById('patientId').value);
        
        // Make prediction request
        const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
            headers: { 
                'Authorization': `Bearer ${API_TOKEN}`,
                'Content-Type': 'multipart/form-data'
            }
        });
        
        // Handle successful prediction
        currentPrediction = response.data;
        currentPatientId = document.getElementById('patientId').value;
        
        displayResults(currentPrediction);
        
    } catch (error) {
        console.error('Prediction failed:', error);
        showNotification('Prediction failed. Please check your inputs and try again.', 'error');
    } finally {
        hideLoadingState();
    }
}

// Validate form inputs
function validateForm() {
    const required = ['patientId', 'age', 'gender', 'education', 'mmse', 'cdr', 'adasCog', 'faq'];
    const missing = [];
    
    required.forEach(field => {
        const element = document.getElementById(field);
        if (!element.value.trim()) {
            missing.push(field);
            element.classList.add('border-red-500');
        } else {
            element.classList.remove('border-red-500');
        }
    });
    
    const mriFile = document.getElementById('mriFile').files[0];
    const petFile = document.getElementById('petFile').files[0];
    
    if (!mriFile) {
        missing.push('MRI file');
    }
    if (!petFile) {
        missing.push('PET file');
    }
    
    if (missing.length > 0) {
        showNotification(`Please fill in: ${missing.join(', ')}`, 'error');
        return false;
    }
    
    return true;
}

// Display prediction results
function displayResults(results) {
    const resultsPanel = document.getElementById('resultsPanel');
    const predictionLabel = document.getElementById('predictionLabel');
    const confidenceScore = document.getElementById('confidenceScore');
    const uncertaintyScore = document.getElementById('uncertaintyScore');
    const recommendations = document.getElementById('recommendations');
    
    // Set prediction text
    predictionLabel.textContent = results.prediction;
    confidenceScore.textContent = `Confidence: ${(results.confidence * 100).toFixed(1)}%`;
    uncertaintyScore.textContent = `Uncertainty: ${(results.uncertainty * 100).toFixed(1)}%`;
    recommendations.textContent = results.recommendation;
    
    // Show results panel
    resultsPanel.classList.remove('hidden');
    
    // Create charts
    createRiskChart(results.risk_score);
    createProbabilityChart(results.probabilities);
    
    // Scroll to results
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Update recent predictions
    addToRecentPredictions(results);
}

// Create risk assessment chart
function createRiskChart(riskScore) {
    const data = [{
        type: "indicator",
        mode: "gauge+number",
        value: riskScore * 100,
        domain: { x: [0, 1], y: [0, 1] },
        title: { text: "Risk Score (%)" },
        gauge: {
            axis: { range: [null, 100] },
            bar: { color: riskScore > 0.7 ? "#ef4444" : riskScore > 0.4 ? "#f59e0b" : "#10b981" },
            steps: [
                { range: [0, 40], color: "#d1fae5" },
                { range: [40, 70], color: "#fef3c7" },
                { range: [70, 100], color: "#fee2e2" }
            ],
            threshold: {
                line: { color: "red", width: 4 },
                thickness: 0.75,
                value: 90
            }
        }
    }];

    const layout = {
        width: 300,
        height: 200,
        margin: { t: 25, b: 25, l: 25, r: 25 },
        paper_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#374151", family: "Arial" }
    };

    Plotly.newPlot('riskChart', data, layout, { displayModeBar: false });
}

// Create probability distribution chart
function createProbabilityChart(probabilities) {
    const labels = ['Cognitively Normal', 'Mild Cognitive Impairment', 'Alzheimer\'s Disease'];
    const values = [
        probabilities.Cognitively_Normal || probabilities['Cognitively_Normal'],
        probabilities.Mild_Cognitive_Impairment || probabilities['Mild_Cognitive_Impairment'],
        probabilities.Alzheimers_Disease || probabilities['Alzheimers_Disease']
    ];

    const data = [{
        x: labels,
        y: values.map(v => v * 100),
        type: 'bar',
        marker: {
            color: ['#10b981', '#f59e0b', '#ef4444'],
            line: { color: '#374151', width: 1 }
        },
        text: values.map(v => (v * 100).toFixed(1) + '%'),
        textposition: 'auto'
    }];

    const layout = {
        title: {
            text: 'Prediction Probabilities',
            font: { size: 16, color: '#374151' }
        },
        xaxis: { 
            title: 'Diagnosis',
            tickangle: -45,
            font: { color: '#374151' }
        },
        yaxis: { 
            title: 'Probability (%)',
            range: [0, 100],
            font: { color: '#374151' }
        },
        margin: { t: 50, b: 100, l: 60, r: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "Arial" }
    };

    Plotly.newPlot('probabilityChart', data, layout, { displayModeBar: false });
}

// Load performance metrics
async function loadPerformanceMetrics() {
    try {
        // Create performance chart
        const data = [{
            values: [94, 3, 3],
            labels: ['Correct Predictions', 'False Positives', 'False Negatives'],
            type: 'pie',
            marker: {
                colors: ['#10b981', '#f59e0b', '#ef4444']
            },
            textinfo: 'label+percent',
            textposition: 'auto'
        }];

        const layout = {
            title: {
                text: 'Model Accuracy (Last 30 Days)',
                font: { size: 14, color: '#374151' }
            },
            margin: { t: 40, b: 20, l: 20, r: 20 },
            paper_bgcolor: "rgba(0,0,0,0)",
            font: { family: "Arial", size: 12, color: '#374151' },
            showlegend: false
        };

        Plotly.newPlot('performanceChart', data, layout, { displayModeBar: false });
        
    } catch (error) {
        console.error('Failed to load performance metrics:', error);
    }
}

// Load recent predictions
function loadRecentPredictions() {
    const recentPredictions = [
        { patientId: 'PAT-001', prediction: 'Cognitively Normal', confidence: 0.92, time: '2 hours ago' },
        { patientId: 'PAT-002', prediction: 'Mild Cognitive Impairment', confidence: 0.87, time: '4 hours ago' },
        { patientId: 'PAT-003', prediction: 'Alzheimer\'s Disease', confidence: 0.94, time: '6 hours ago' },
        { patientId: 'PAT-004', prediction: 'Cognitively Normal', confidence: 0.89, time: '8 hours ago' }
    ];

    const container = document.getElementById('recentPredictions');
    container.innerHTML = '';

    recentPredictions.forEach(pred => {
        const div = document.createElement('div');
        div.className = 'flex items-center justify-between p-3 bg-gray-50 rounded-md hover:bg-gray-100 cursor-pointer';
        div.innerHTML = `
            <div>
                <p class="font-medium text-gray-800">${pred.patientId}</p>
                <p class="text-sm text-gray-600">${pred.prediction}</p>
            </div>
            <div class="text-right">
                <p class="text-sm font-medium text-green-600">${(pred.confidence * 100).toFixed(0)}%</p>
                <p class="text-xs text-gray-500">${pred.time}</p>
            </div>
        `;
        container.appendChild(div);
    });
}

// Add to recent predictions
function addToRecentPredictions(results) {
    const container = document.getElementById('recentPredictions');
    const div = document.createElement('div');
    div.className = 'flex items-center justify-between p-3 bg-blue-50 rounded-md border border-blue-200';
    div.innerHTML = `
        <div>
            <p class="font-medium text-gray-800">${currentPatientId}</p>
            <p class="text-sm text-gray-600">${results.prediction}</p>
        </div>
        <div class="text-right">
            <p class="text-sm font-medium text-green-600">${(results.confidence * 100).toFixed(0)}%</p>
            <p class="text-xs text-gray-500">Just now</p>
        </div>
    `;
    
    // Insert at the beginning
    container.insertBefore(div, container.firstChild);
    
    // Remove last item if more than 5
    if (container.children.length > 5) {
        container.removeChild(container.lastChild);
    }
}

// Show loading state
function showLoadingState() {
    const submitButton = document.querySelector('button[type="submit"]');
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Analyzing...';
    submitButton.disabled = true;
}

// Hide loading state
function hideLoadingState() {
    const submitButton = document.querySelector('button[type="submit"]');
    submitButton.innerHTML = '<i class="fas fa-brain mr-2"></i>Analyze Patient';
    submitButton.disabled = false;
}

// Clear form
function clearForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('mriFileName').textContent = '';
    document.getElementById('petFileName').textContent = '';
    document.getElementById('resultsPanel').classList.add('hidden');
}

// Generate report
function generateReport() {
    if (!currentPrediction) {
        showNotification('No prediction data available', 'error');
        return;
    }
    
    // In a real implementation, this would generate a PDF report
    showNotification('Report generation feature coming soon', 'info');
}

// View explanation
function viewExplanation() {
    if (!currentPatientId) {
        showNotification('No patient data available', 'error');
        return;
    }
    
    // In a real implementation, this would show model explanations
    showNotification('Model explanation feature coming soon', 'info');
}

// Add feedback
function addFeedback() {
    if (!currentPatientId) {
        showNotification('No patient data available', 'error');
        return;
    }
    
    // In a real implementation, this would open a feedback modal
    showNotification('Feedback feature coming soon', 'info');
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 px-6 py-3 rounded-md text-white z-50 ${
        type === 'error' ? 'bg-red-500' : 
        type === 'warning' ? 'bg-yellow-500' : 
        type === 'success' ? 'bg-green-500' : 'bg-blue-500'
    }`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Animate in
    notification.style.transform = 'translateX(100%)';
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
        notification.style.transition = 'transform 0.3s ease';
    }, 100);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Export functions for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initializeDashboard,
        validateForm,
        displayResults,
        createRiskChart,
        createProbabilityChart
    };
}