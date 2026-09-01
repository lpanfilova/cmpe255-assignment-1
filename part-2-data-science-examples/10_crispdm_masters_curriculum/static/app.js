const questions = [
  {q:"Which CRISP-DM phase should define the decision and success metric?", a:["Business understanding","Modeling","Deployment"], c:0, e:"The analytical target must serve a defined decision."},
  {q:"Why put imputation inside the supervised pipeline?", a:["It makes plots prettier","It prevents test information leaking into training","It removes every outlier"], c:1, e:"Fold-specific preprocessing preserves honest evaluation."},
  {q:"A silhouette score measures…", a:["cluster cohesion and separation","classification calibration","causal effect"], c:0, e:"It compares within-cluster closeness with neighboring-cluster separation."},
  {q:"An Isolation Forest flag proves the row is bad data.", a:["True","False"], c:1, e:"It only marks unusual feature combinations under the fitted policy."},
  {q:"Lift above 1 means…", a:["the items co-occur more than independence predicts","the rule is causal","support is 100%"], c:0, e:"Lift compares observed confidence with baseline prevalence."},
  {q:"Why can LSH be faster than a full scan?", a:["It sorts labels","It hashes likely neighbors into shared buckets","It trains a classifier"], c:1, e:"Only bucket candidates receive exact similarity calculations."},
  {q:"Which metric measures ranking quality across thresholds?", a:["ROC-AUC","row count","support"], c:0, e:"ROC-AUC is threshold-independent ranking discrimination."}
];
const root=document.querySelector('#quiz-root');
questions.forEach((item,i)=>{const card=document.createElement('article');card.className='question';card.innerHTML=`<h3>${i+1}. ${item.q}</h3>`;item.a.forEach((answer,j)=>{const button=document.createElement('button');button.textContent=answer;button.onclick=()=>{card.querySelectorAll('button').forEach(b=>b.disabled=true);card.classList.add(j===item.c?'correct':'wrong');const p=document.createElement('p');p.textContent=(j===item.c?'Correct. ':'Not quite. ')+item.e;card.appendChild(p)};card.appendChild(button)});root.appendChild(card)});

