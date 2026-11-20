/* Canvas input, live training updates, and a faithful compact network diagram. */
const canvas = document.querySelector('#drawing-canvas');
const context = canvas.getContext('2d');
const prediction = document.querySelector('#prediction');
const scores = document.querySelector('#scores');
const trainButton = document.querySelector('#train-button');
const networkView = document.querySelector('#network-view');
let drawing = false;

context.strokeStyle = 'white'; context.lineWidth = 18; context.lineCap = 'round'; context.lineJoin = 'round';
function clearCanvas() { context.fillStyle = 'black'; context.fillRect(0, 0, canvas.width, canvas.height); prediction.textContent = 'Draw one digit.'; scores.innerHTML = ''; }
function point(event) { const bounds = canvas.getBoundingClientRect(); return { x: (event.clientX - bounds.left) * canvas.width / bounds.width, y: (event.clientY - bounds.top) * canvas.height / bounds.height }; }
function draw(event) { if (!drawing) return; const current = point(event); context.lineTo(current.x, current.y); context.stroke(); }
canvas.addEventListener('pointerdown', event => { drawing = true; canvas.setPointerCapture(event.pointerId); const current = point(event); context.beginPath(); context.moveTo(current.x, current.y); });
canvas.addEventListener('pointermove', draw); canvas.addEventListener('pointerup', () => drawing = false); canvas.addEventListener('pointercancel', () => drawing = false);

function mnistPixels() {
  const source = context.getImageData(0, 0, 280, 280); let minX = 280, minY = 280, maxX = -1, maxY = -1;
  for (let y = 0; y < 280; y++) for (let x = 0; x < 280; x++) if (source.data[(y * 280 + x) * 4] > 20) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
  if (maxX < 0) throw new Error('Please draw a digit first.');
  const padding = 12; minX = Math.max(0, minX - padding); minY = Math.max(0, minY - padding); maxX = Math.min(279, maxX + padding); maxY = Math.min(279, maxY + padding);
  const crop = document.createElement('canvas'); crop.width = maxX - minX + 1; crop.height = maxY - minY + 1; crop.getContext('2d').putImageData(source, -minX, -minY);
  const small = document.createElement('canvas'); small.width = small.height = 28; const smallContext = small.getContext('2d'); const scale = Math.min(20 / crop.width, 20 / crop.height); const width = crop.width * scale, height = crop.height * scale;
  smallContext.drawImage(crop, (28 - width) / 2, (28 - height) / 2, width, height); const pixels = smallContext.getImageData(0, 0, 28, 28).data;
  return Array.from({ length: 784 }, (_, index) => pixels[index * 4] / 255);
}

function colourForWeight(weight, maximum) { const strength = Math.min(Math.abs(weight) / (maximum || 1), 1); return { colour: weight >= 0 ? '#2166e8' : '#e14628', opacity: 0.12 + strength * 0.72, width: 0.35 + strength * 1.7 }; }
function maxMagnitude(matrix) { return Math.max(...matrix.flat().map(Math.abs), 0.001); }
function node(x, y, label, activity, radius = 10) { const intensity = Math.min(activity || 0, 1); const fill = intensity ? `rgb(${Math.round(255 - 190 * intensity)}, ${Math.round(255 - 92 * intensity)}, ${Math.round(255 - 151 * intensity)})` : '#fff'; return `<circle cx="${x}" cy="${y}" r="${radius}" fill="${fill}" stroke="#111"/><text class="node-label" x="${x}" y="${y}">${label}</text>`; }
function gridPositions() { return Array.from({ length: 16 }, (_, index) => ({ x: 72 + (index % 4) * 24, y: 120 + Math.floor(index / 4) * 55 })); }
function verticalPositions(count, x) { return Array.from({ length: count }, (_, index) => ({ x, y: 85 + index * (280 / Math.max(count - 1, 1)) })); }

function renderNetwork(data) {
  if (!data) return; const input = gridPositions(); const output = verticalPositions(10, 665); const outputActivity = data.output_activity || Array(10).fill(0);
  let edges = '', nodes = '', labels = `<text class="layer-label" x="108" y="45">INPUT (16 regions)</text><text class="layer-label" x="665" y="45">OUTPUT (0–9)</text>`;
  if (data.mode === 'expanded') {
    const hidden = verticalPositions(data.hidden_size, 385); const maxA = maxMagnitude(data.input_to_hidden); const maxB = maxMagnitude(data.hidden_to_output);
    data.input_to_hidden.forEach((row, source) => row.forEach((weight, target) => { const style = colourForWeight(weight, maxA); edges += `<line class="edge" x1="${input[source].x}" y1="${input[source].y}" x2="${hidden[target].x}" y2="${hidden[target].y}" stroke="${style.colour}" stroke-opacity="${style.opacity}" stroke-width="${style.width}"/>`; }));
    data.hidden_to_output.forEach((row, source) => row.forEach((weight, target) => { const style = colourForWeight(weight, maxB); edges += `<line class="edge" x1="${hidden[source].x}" y1="${hidden[source].y}" x2="${output[target].x}" y2="${output[target].y}" stroke="${style.colour}" stroke-opacity="${style.opacity}" stroke-width="${style.width}"/>`; }));
    nodes += hidden.map((item, index) => node(item.x, item.y, index + 1, data.hidden_activity[index], 8)).join(''); labels += `<text class="layer-label" x="385" y="45">HIDDEN (${data.hidden_size})</text>`;
    document.querySelector('#network-description').textContent = `Expanded view: all ${data.hidden_size} hidden neurons and their current learned connections.`;
  } else {
    const maxWeight = maxMagnitude(data.input_to_output);
    data.input_to_output.forEach((row, source) => row.forEach((weight, target) => { const style = colourForWeight(weight, maxWeight); edges += `<line class="edge" x1="${input[source].x}" y1="${input[source].y}" x2="${output[target].x}" y2="${output[target].y}" stroke="${style.colour}" stroke-opacity="${style.opacity}" stroke-width="${style.width}"/>`; }));
    document.querySelector('#network-description').textContent = `Compact view: hidden layer has ${data.hidden_size} neurons, so it is collapsed into its real combined input-to-output influence.`;
  }
  nodes += input.map((item, index) => node(item.x, item.y, '', data.input_activity[index], 8)).join(''); nodes += output.map((item, index) => node(item.x, item.y, index, outputActivity[index], 12)).join('');
  networkView.innerHTML = `<svg class="network-svg" viewBox="0 0 730 410" role="img" aria-label="Network visualization">${labels}${edges}${nodes}</svg>`;
}

async function predictDigit() { try { prediction.textContent = 'Predicting…'; const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pixels: mnistPixels() }) }); const data = await response.json(); if (!response.ok) throw new Error(data.error); prediction.innerHTML = `Prediction: <strong>${data.prediction}</strong> / ${data.confidence}%`; scores.innerHTML = data.probabilities.map((score, digit) => `<span class="score ${digit === data.prediction ? 'winner' : ''}">${digit}<br>${score}%</span>`).join(''); renderNetwork(data.visualization); } catch (error) { prediction.textContent = error.message; } }
async function updateTrainingStatus() { const data = await (await fetch('/api/training-status')).json(); document.querySelector('#training-message').textContent = data.message; trainButton.disabled = data.running; document.querySelector('#history').innerHTML = data.history.map(item => `<div>Epoch ${item.epoch} — loss ${item.loss} — test ${(item.accuracy * 100).toFixed(2)}%</div>`).join(''); if (data.visualization) renderNetwork(data.visualization); }
trainButton.addEventListener('click', async () => { const response = await fetch('/api/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ learning_rate: document.querySelector('#learning-rate').value, epochs: document.querySelector('#epochs').value, hidden_size: document.querySelector('#hidden-size').value }) }); const data = await response.json(); document.querySelector('#training-message').textContent = data.message || data.error; updateTrainingStatus(); });
document.querySelector('#predict-button').addEventListener('click', predictDigit); document.querySelector('#clear-button').addEventListener('click', clearCanvas); clearCanvas(); fetch('/api/model-visualization').then(response => response.json()).then(renderNetwork); setInterval(updateTrainingStatus, 700);
