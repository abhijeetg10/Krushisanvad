// --- CONFIGURATION ---
const BACKEND_URL = "http://127.0.0.1:5000"; 

// --- PRELOADER LOGIC ---
window.onload = function() {
  document.getElementById('preloader').style.opacity = '0';
  setTimeout(() => { document.getElementById('preloader').style.display = 'none'; }, 500);
};

// --- CORE FUNCTION: TRIGGERED BY BUTTON CLICK ---
async function runAnalysis() {
  
  // 1. Get the file directly
  const fileInput = document.getElementById('pdfFile');
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select a Soil Health Card PDF first.");
    return;
  }

  // 2. UI State: Switch to Loading View
  document.getElementById('placeholderView').style.display = 'none';
  document.getElementById('resultView').style.display = 'none';
  document.getElementById('loadingView').style.display = 'block';

  // 3. Prepare Data
  const formData = new FormData();
  formData.append('file', file);

  try {
    // 4. Send POST Request
    console.log("📤 Sending PDF to Backend...");
    const response = await fetch(`${BACKEND_URL}/analyze`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error("Analysis failed. Ensure the backend is running.");
    }
    
    // 5. Parse JSON Response
    const data = await response.json();
    console.log("✅ Data Received:", data);

    // 6. Update UI
    updateDashboard(data);

  } catch (error) {
    console.error("❌ Error:", error);
    alert("Analysis Error: " + error.message);
    document.getElementById('loadingView').style.display = 'none';
    document.getElementById('placeholderView').style.display = 'block';
  }
}

// --- HELPER: UPDATE HTML WITH JSON DATA ---
function updateDashboard(data) {
  // A. Fill Extracted Soil Data
  document.getElementById('val_N').innerText = (data.soil_data.N || 0).toFixed(1) + " kg/ha";
  document.getElementById('val_P').innerText = (data.soil_data.P || 0).toFixed(1) + " kg/ha";
  document.getElementById('val_K').innerText = (data.soil_data.K || 0).toFixed(1) + " kg/ha";
  document.getElementById('val_ph').innerText = (data.soil_data.ph || 7.0).toFixed(1);
  document.getElementById('val_ec').innerText = (data.soil_data.EC || 0).toFixed(2) + " dS/m";
  document.getElementById('val_oc').innerText = (data.soil_data.OC || 0).toFixed(2) + " %";

  // B. Fill Crop Prediction
  document.getElementById('cropName').innerText = data.crop.prediction;
  document.getElementById('cropConfText').innerText = data.crop.confidence + "% Match";
  document.getElementById('cropConfBar').style.width = data.crop.confidence + "%";

  // C. Fill Explanations
  const reasonList = document.getElementById('cropReasons');
  reasonList.innerHTML = ""; 
  if (data.crop.reasons && data.crop.reasons.length > 0) {
    data.crop.reasons.forEach(reason => {
      const li = document.createElement('li');
      li.innerText = reason;
      li.style.marginBottom = "8px"; 
      reasonList.appendChild(li);
    });
  } else {
    reasonList.innerHTML = "<li>AI analysis complete based on nutrient profile.</li>";
  }

  // D. Fill Fertilizer Recommendation
  document.getElementById('fertName').innerText = data.fertilizer.prediction;
  document.getElementById('fertConf').innerText = data.fertilizer.confidence;

  // E. Load SHAP Plot
  // We append a timestamp to force the browser to reload the image
  const plotUrl = `${BACKEND_URL}${data.crop.plot}`;
  document.getElementById('shapFrame').src = `${plotUrl}?t=${new Date().getTime()}`;

  // F. Show Results View
  document.getElementById('loadingView').style.display = 'none';
  document.getElementById('resultView').style.display = 'block';
  
  // Smooth scroll to result
  document.getElementById('resultView').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// --- UTILS ---
function openBookingModal() { document.getElementById('bookingModal').style.display = 'flex'; }
function closeBookingModal() { document.getElementById('bookingModal').style.display = 'none'; }

function toggleChat() {
  const chat = document.getElementById('chatWindow');
  chat.style.display = chat.style.display === 'flex' ? 'none' : 'flex';
}

function sendMessage() {
  const input = document.getElementById('chatInput');
  const body = document.getElementById('chatBody');
  const text = input.value.trim();
  
  if (text) {
    const userMsg = document.createElement('div');
    userMsg.className = 'message user';
    userMsg.innerText = text;
    body.appendChild(userMsg);
    input.value = '';
    body.scrollTop = body.scrollHeight;

    setTimeout(() => {
      const botMsg = document.createElement('div');
      botMsg.className = 'message bot';
      botMsg.innerText = "That's a great question! Based on your soil report, I recommend focusing on maintaining pH levels between 6.5 and 7.5.";
      body.appendChild(botMsg);
      body.scrollTop = body.scrollHeight;
    }, 1000);
  }
}