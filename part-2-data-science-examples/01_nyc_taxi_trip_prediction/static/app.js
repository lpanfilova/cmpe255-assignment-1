const map = L.map('map', { zoomControl: false }).setView([40.758, -73.9855], 12);
L.control.zoom({ position: 'bottomright' }).addTo(map);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '' }).addTo(map);
const colors = ['#ff5c35', '#262a33'];
let points = [], markers = [], line;

const timeInput = document.querySelector('#pickupTime');
const now = new Date(); now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
timeInput.value = now.toISOString().slice(0, 16);

function markerIcon(index) {
  return L.divIcon({ className: '', html: `<div class="pin" style="background:${colors[index]}">${index ? 'B' : 'A'}</div>`, iconSize: [34, 34], iconAnchor: [17, 17] });
}
function refresh() {
  ['pickupText', 'dropoffText'].forEach((id, i) => document.querySelector('#' + id).textContent = points[i] ? `${points[i].lat.toFixed(4)}, ${points[i].lng.toFixed(4)}` : 'Choose on map');
  document.querySelector('#estimate').disabled = points.length !== 2;
  document.querySelector('#hint').textContent = points.length === 0 ? 'Click once for pickup · again for drop-off' : points.length === 1 ? 'Now choose the drop-off' : 'Ready to estimate';
}
map.on('click', ({ latlng }) => {
  if (points.length === 2) reset();
  const index = points.length; points.push(latlng);
  markers.push(L.marker(latlng, { icon: markerIcon(index) }).addTo(map));
  if (points.length === 2) line = L.polyline(points, { color: '#ff5c35', weight: 3, dashArray: '8 8' }).addTo(map);
  refresh();
});
function reset() { markers.forEach(m => map.removeLayer(m)); if (line) map.removeLayer(line); points = []; markers = []; line = null; refresh(); }
document.querySelector('#reset').addEventListener('click', reset);
document.querySelector('#estimate').addEventListener('click', async () => {
  const button = document.querySelector('#estimate'); button.disabled = true; button.textContent = 'Estimating…';
  const payload = { pickup_latitude: points[0].lat, pickup_longitude: points[0].lng, dropoff_latitude: points[1].lat, dropoff_longitude: points[1].lng, pickup_datetime: timeInput.value, passenger_count: +document.querySelector('#passengers').value };
  try {
    const response = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
    const data = await response.json(); if (!response.ok) throw new Error(data.error);
    document.querySelector('#result').innerHTML = `<small>ESTIMATED DURATION</small><strong>${data.duration_minutes} <i>min</i></strong><p>Approx. ${data.estimated_route_km} km by road · ${data.straight_line_km} km direct</p>`;
  } catch (error) { document.querySelector('#result').innerHTML = `<small>COULD NOT ESTIMATE</small><p>${error.message}</p>`; }
  finally { button.disabled = false; button.innerHTML = 'Estimate trip <span>→</span>'; }
});

