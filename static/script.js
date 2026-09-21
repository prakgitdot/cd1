const form = document.getElementById('crop-form');
const submitBtn = document.getElementById('submit-btn');
const resultArea = document.getElementById('result-area');
const resultCrop = document.getElementById('result-crop');
const resultAltsList = document.getElementById('result-alts-list');
const errorArea = document.getElementById('error-area');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  errorArea.hidden = true;
  resultArea.hidden = true;
  submitBtn.disabled = true;
  submitBtn.textContent = 'Checking the plot…';

  const payload = {
    N: document.getElementById('N').value,
    P: document.getElementById('P').value,
    K: document.getElementById('K').value,
    temperature: document.getElementById('temperature').value,
    humidity: document.getElementById('humidity').value,
    ph: document.getElementById('ph').value,
    rainfall: document.getElementById('rainfall').value,
  };

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || 'Something went wrong reading the model.');
    }

    resultCrop.textContent = data.recommended_crop;

    resultAltsList.innerHTML = '';
    data.top_3.forEach((item) => {
      const li = document.createElement('li');
      li.textContent = item.crop;
      const span = document.createElement('span');
      span.textContent = `${item.confidence}%`;
      li.appendChild(span);
      resultAltsList.appendChild(li);
    });

    resultArea.hidden = false;
  } catch (err) {
    errorArea.textContent = err.message;
    errorArea.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Recommend a crop';
  }
});
