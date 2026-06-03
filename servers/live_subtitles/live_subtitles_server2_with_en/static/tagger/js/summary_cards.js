// ===== Summary Cards =====
function updateSummaryCards(stats, chunks, segments, isFiltered) {
  if (isFiltered && chunks.length===1) {
    const s = chunks[0];
    document.getElementById("cardSegments").textContent="1";
    document.getElementById("cardTotal").textContent="1";
    document.getElementById("cardSpeech").textContent=s.speech_detected?"✅ Yes":"❌ No";
    document.getElementById("cardSpeechPct").textContent="Single chunk view";
    document.getElementById("cardAvgChunks").textContent="1";
    document.getElementById("cardAvgProb").textContent=(s.speech_probability||0).toFixed(3);
    if (s.top_predictions&&s.top_predictions.length>0) {
      document.getElementById("cardTopEvent").textContent=s.top_predictions[0].name;
      document.getElementById("cardTopEventProb").textContent=`${(s.top_predictions[0].prob*100).toFixed(1)}% probability`;
    } else { document.getElementById("cardTopEvent").textContent="N/A"; document.getElementById("cardTopEventProb").textContent="No predictions"; }
    return;
  }
  const sc = segments?segments.length:0, cc = chunks.length;
  document.getElementById("cardSegments").textContent=sc;
  document.getElementById("cardTotal").textContent=cc;
  const speechSegs = segments?segments.filter(s=>s.chunks.some(c=>c.speech_detected)).length:chunks.filter(c=>c.speech_detected).length;
  document.getElementById("cardSpeech").textContent=speechSegs;
  document.getElementById("cardSpeechPct").textContent=sc>0?`${((speechSegs/sc)*100).toFixed(1)}% of segments`:"-";
  document.getElementById("cardAvgChunks").textContent=sc>0?(cc/sc).toFixed(1):"-";
  const probs = chunks.map(c=>c.speech_probability).filter(p=>p!=null);
  document.getElementById("cardAvgProb").textContent=probs.length>0?(probs.reduce((a,b)=>a+b,0)/probs.length).toFixed(3):"-";
  const te = getTopEventNames(chunks,1);
  if (te.length>0) {
    document.getElementById("cardTopEvent").textContent=te[0].name;
    document.getElementById("cardTopEventProb").textContent=`${(te[0].avgProb*100).toFixed(1)}% avg (${te[0].count} chunks)`;
  } else { document.getElementById("cardTopEvent").textContent="N/A"; document.getElementById("cardTopEventProb").textContent="No data"; }
}