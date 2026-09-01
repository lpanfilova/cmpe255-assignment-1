'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const fmt = (x, digits = 2) => Number(x).toFixed(digits);

// Course progress is local-only and intentionally survives refreshes.
const visited = new Set(JSON.parse(localStorage.getItem('signal-lab-progress') || '[]'));
function renderProgress() {
  $('#progressLabel').textContent = `${visited.size} / 4 explored`;
  $('#progressBar').style.width = `${visited.size * 25}%`;
}
const observer = new IntersectionObserver(entries => entries.forEach(entry => {
  if (entry.isIntersecting) {
    visited.add(entry.target.dataset.module);
    localStorage.setItem('signal-lab-progress', JSON.stringify([...visited]));
    renderProgress();
  }
}), { threshold: 0.3 });
$$('[data-module]').forEach(section => observer.observe(section));
$('#resetProgress').addEventListener('click', () => { visited.clear(); localStorage.removeItem('signal-lab-progress'); renderProgress(); });
renderProgress();

// Hero: a small animated loss landscape.
const hero = $('#heroCanvas'), hctx = hero.getContext('2d');
let heroT = 0;
function drawHero() {
  const w = hero.width, h = hero.height; hctx.clearRect(0, 0, w, h);
  hctx.strokeStyle = 'rgba(231,244,233,.18)'; hctx.lineWidth = 1;
  for (let r = 45; r < 250; r += 36) { hctx.beginPath(); hctx.ellipse(w / 2, h / 2 - 5, r * 1.25, r * .66, -.18, 0, Math.PI * 2); hctx.stroke(); }
  for (let i = 0; i < 38; i++) {
    const positive = i % 2 === 0, a = i * 2.399, radius = 60 + (i * 37) % 185;
    const x = w / 2 + Math.cos(a) * radius * 1.25, y = h / 2 + Math.sin(a) * radius * .65;
    hctx.fillStyle = positive ? 'rgba(200,242,91,.82)' : 'rgba(255,150,126,.75)';
    hctx.beginPath(); positive ? hctx.arc(x, y, 4.5, 0, 7) : hctx.rect(x - 4, y - 4, 8, 8); hctx.fill();
  }
  const decay = Math.exp(-(heroT % 300) / 105), angle = heroT * .035;
  const x = w / 2 + Math.cos(angle) * 215 * decay, y = h / 2 + Math.sin(angle) * 115 * decay;
  hctx.strokeStyle = '#fff'; hctx.lineWidth = 2; hctx.beginPath();
  for (let k = 0; k < 65; k++) { const d = Math.exp(-k / 22), a = angle - k * .035; const px = w / 2 + Math.cos(a) * 215 * d, py = h / 2 + Math.sin(a) * 115 * d; k ? hctx.lineTo(px, py) : hctx.moveTo(px, py); } hctx.stroke();
  hctx.fillStyle = '#fff'; hctx.beginPath(); hctx.arc(x, y, 8, 0, 7); hctx.fill();
  $('#heroLoss').textContent = `loss ${fmt(.06 + .86 * decay * decay)}`;
  heroT++;
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches) requestAnimationFrame(drawHero);
}
drawHero();

// Naive Bayes: Bernoulli likelihoods for selected evidence.
function updateBayes() {
  const prior = +$('#prior').value / 100; let spam = prior, ham = 1 - prior;
  const selected = $$('.bayes-feature:checked');
  selected.forEach(f => { spam *= +f.dataset.spam; ham *= +f.dataset.ham; });
  const posterior = spam / (spam + ham), pct = Math.round(posterior * 100);
  $('#priorOut').textContent = `${Math.round(prior * 100)}%`; $('#posterior').textContent = `${pct}%`;
  $('#bayesDonut').style.setProperty('--p', pct); $('#spamBar').style.width = `${pct}%`; $('#hamBar').style.width = `${100 - pct}%`;
  $('#bayesVerdict').textContent = posterior >= .5 ? 'Likely spam' : 'Likely ham';
  $('#bayesMath').textContent = `${selected.length} evidence vote${selected.length === 1 ? '' : 's'} · odds ${fmt(spam / Math.max(ham, 1e-9), 1)} : 1`;
}
$('#prior').addEventListener('input', updateBayes); $$('.bayes-feature').forEach(x => x.addEventListener('change', updateBayes));
$('#bayesReset').addEventListener('click', () => { $('#prior').value = 40; $$('.bayes-feature').forEach((x, i) => x.checked = i !== 1); updateBayes(); }); updateBayes();

// Fixed deterministic score sample keeps comparisons reproducible.
const scores = [
  [.94,1],[.88,1],[.83,0],[.79,1],[.74,1],[.69,0],[.65,1],[.61,0],[.58,1],[.54,0],
  [.49,1],[.46,0],[.42,0],[.38,1],[.35,0],[.31,0],[.27,1],[.23,0],[.18,0],[.12,0]
];
function countsAt(t) { let tp=0,fp=0,tn=0,fn=0; scores.forEach(([s,y]) => { if(s>=t) y?tp++:fp++; else y?fn++:tn++; }); return {tp,fp,tn,fn}; }
function rocPoints() { return [...new Set([1.01,...scores.map(x=>x[0]),0])].sort((a,b)=>b-a).map(t => { const c=countsAt(t); return {x:c.fp/(c.fp+c.tn),y:c.tp/(c.tp+c.fn)}; }); }
const roc = rocPoints();
let aucValue = 0; for(let i=1;i<roc.length;i++) aucValue += (roc[i].x-roc[i-1].x)*(roc[i].y+roc[i-1].y)/2;
function drawScores(t) {
  const c=$('#scoreCanvas'),x=c.getContext('2d'),w=c.width,h=c.height;x.clearRect(0,0,w,h);
  x.strokeStyle='#354850';x.beginPath();x.moveTo(35,h-38);x.lineTo(w-25,h-38);x.stroke();
  for(let i=0;i<=10;i++){const px=35+i*(w-60)/10;x.fillStyle='#8da097';x.font='11px monospace';x.fillText((i/10).toFixed(1),px-8,h-18)}
  scores.forEach(([s,y],i)=>{const px=35+s*(w-60),py=y?78+(i%3)*18:155+(i%3)*18;x.fillStyle=y?'#c8f25b':'#ff917f';x.beginPath();x.arc(px,py,7,0,7);x.fill()});
  const tx=35+t*(w-60);x.strokeStyle='#fff';x.setLineDash([5,5]);x.beginPath();x.moveTo(tx,25);x.lineTo(tx,h-40);x.stroke();x.setLineDash([]);x.fillStyle='#fff';x.fillText('threshold',tx-27,16);x.fillStyle='#c8f25b';x.fillText('ACTUAL +',35,48);x.fillStyle='#ff917f';x.fillText('ACTUAL −',35,136);
}
function drawRoc() { const c=$('#rocCanvas'),x=c.getContext('2d'),w=c.width,h=c.height,p=38;x.clearRect(0,0,w,h);x.strokeStyle='#65747b';x.beginPath();x.moveTo(p,h-p);x.lineTo(w-p,p);x.stroke();x.fillStyle='#91a29a';x.font='11px monospace';x.fillText('0',p-12,h-p+15);x.fillText('1',w-p,h-p+15);x.fillText('FPR →',w/2,h-7);x.save();x.translate(11,h/2+20);x.rotate(-Math.PI/2);x.fillText('TPR →',0,0);x.restore();x.strokeStyle='#c8f25b';x.lineWidth=3;x.beginPath();roc.forEach((q,i)=>{const px=p+q.x*(w-2*p),py=h-p-q.y*(h-2*p);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.lineWidth=1; }
function bestCost() { const fpC=+$('#fpCost').value||0,fnC=+$('#fnCost').value||0; let best={cost:Infinity,t:0}; for(let i=0;i<=100;i++){const c=countsAt(i/100),cost=c.fp*fpC+c.fn*fnC;if(cost<best.cost)best={cost,t:i/100}} return best; }
function updateEvaluation(){const t=+$('#threshold').value/100,c=countsAt(t);$('#thresholdOut').textContent=fmt(t);['tp','fp','tn','fn'].forEach(k=>$('#'+k).textContent=c[k]);$('#precision').textContent=fmt(c.tp/Math.max(1,c.tp+c.fp));$('#recall').textContent=fmt(c.tp/(c.tp+c.fn));$('#fpr').textContent=fmt(c.fp/(c.fp+c.tn));$('#accuracy').textContent=fmt((c.tp+c.tn)/scores.length);const cost=c.fp*(+$('#fpCost').value||0)+c.fn*(+$('#fnCost').value||0),best=bestCost();$('#totalCost').textContent=cost;$('#bestThreshold').textContent=`Lowest sampled cost: ${best.cost} at threshold ${fmt(best.t)}`;drawScores(t)}
['threshold','fpCost','fnCost'].forEach(id=>$('#'+id).addEventListener('input',updateEvaluation)); $('#auc').textContent=fmt(aucValue,3);drawRoc();updateEvaluation();

// Gradient descent on L(w)=(w-2)^2+0.35.
const dc=$('#descentCanvas'),dctx=dc.getContext('2d');let theta=-3.2,history=[theta],autoTimer=null;
const loss=w=>(w-2)**2+.35, grad=w=>2*(w-2);
function toDX(w){return 45+(w+4)*(dc.width-75)/8}
function toDY(y){return dc.height-35-y*(dc.height-65)/38}
function drawDescent(){dctx.clearRect(0,0,dc.width,dc.height);dctx.strokeStyle='#d9ddd7';dctx.beginPath();dctx.moveTo(35,dc.height-35);dctx.lineTo(dc.width-20,dc.height-35);dctx.stroke();dctx.strokeStyle='#1e6848';dctx.lineWidth=3;dctx.beginPath();for(let i=0;i<=160;i++){const w=-4+i/20,x=toDX(w),y=toDY(loss(w));i?dctx.lineTo(x,y):dctx.moveTo(x,y)}dctx.stroke();dctx.lineWidth=1;history.forEach((w,i)=>{dctx.fillStyle=i===history.length-1?'#ff725e':'rgba(255,114,94,.25)';dctx.beginPath();dctx.arc(toDX(w),toDY(loss(w)),i===history.length-1?8:4,0,7);dctx.fill()});dctx.fillStyle='#68716d';dctx.font='12px monospace';dctx.fillText('parameter θ',dc.width-105,dc.height-10);dctx.fillText('minimum θ = 2',toDX(2)-43,toDY(.35)-16)}
function descentStep(){const eta=+$('#learningRate').value/100,g=grad(theta),next=theta-eta*g;$('#stepEquation').textContent=`θ ← ${fmt(theta)} − ${fmt(eta)} × (${fmt(g)}) = ${fmt(next)} · loss ${fmt(loss(next))}`;theta=clamp(next,-4,4);history.push(theta);if(history.length>30)history.shift();drawDescent()}
$('#learningRate').addEventListener('input',e=>$('#lrOut').textContent=fmt(+e.target.value/100));$('#descentStep').addEventListener('click',descentStep);$('#descentReset').addEventListener('click',()=>{theta=-3.2;history=[theta];$('#stepEquation').textContent='';drawDescent()});$('#descentAuto').addEventListener('click',()=>{if(autoTimer){clearInterval(autoTimer);autoTimer=null;$('#descentAuto').textContent='Auto-run'}else{autoTimer=setInterval(descentStep,450);$('#descentAuto').textContent='Pause'}});drawDescent();

// One sigmoid neuron with binary cross-entropy, x=2 and target y=1.
const sigmoid=z=>1/(1+Math.exp(-z));
function networkValues(){const w=+$('#weight').value/100,x=2,y=1,z=w*x,yhat=sigmoid(z),L=-(y*Math.log(yhat)+(1-y)*Math.log(1-yhat)),dLdy=-(y/yhat)+(1-y)/(1-yhat),dydz=yhat*(1-yhat),dzdw=x,dLdw=dLdy*dydz*dzdw;return{w,z,yhat,L,dLdy,dydz,dzdw,dLdw}}
function updateNetwork(){const v=networkValues();$('#weightOut').textContent=fmt(v.w);$('#zValue').textContent=`z = ${fmt(v.z)}`;$('#yhatValue').textContent=`ŷ = ${fmt(v.yhat)}`;$('#lossValue').textContent=`L = ${fmt(v.L)}`;$('#gradLoss').textContent=fmt(v.dLdy);$('#gradSigmoid').textContent=fmt(v.dydz);$('#gradLinear').textContent=fmt(v.dzdw);$('#gradWeight').textContent=fmt(v.dLdw);$('#networkNarrative').textContent=`${fmt(v.dLdy)} × ${fmt(v.dydz)} × ${fmt(v.dzdw)} = ${fmt(v.dLdw)}. The negative gradient says increasing w will reduce loss.`}
$('#weight').addEventListener('input',updateNetwork);$('#networkReset').addEventListener('click',()=>{$('#weight').value=-50;updateNetwork()});$('#applyGradient').addEventListener('click',()=>{const v=networkValues();$('#weight').value=Math.round(clamp(v.w-.5*v.dLdw,-2,2)*100);updateNetwork()});updateNetwork();

// Quizzes: one attempt can be revised; explanations reinforce the concept.
const explanations={bayes:'Correct: logs prevent floating-point underflow and turn products into sums.',evaluation:'Correct: the actual condition is positive, but the prediction is negative—a false negative (Type II error).',calculus:'Correct: gradient descent moves opposite the derivative, so a positive slope means step left.',backprop:'Correct: each local sensitivity scales the upstream effect; their product is the total sensitivity.'};
$$('[data-quiz]').forEach(quiz=>$$('.answers button',quiz).forEach(button=>button.addEventListener('click',()=>{const correct=button.dataset.correct==='true';$$('.answers button',quiz).forEach(b=>b.classList.remove('correct','wrong'));button.classList.add(correct?'correct':'wrong');$('.feedback',quiz).textContent=correct?explanations[quiz.dataset.quiz]:'Not quite—trace what each term or outcome means, then try again.';})));
