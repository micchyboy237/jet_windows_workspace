// ===== Events Timeline =====
function createEventsTimeline(chunks, topEventNames) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const times = chunks.map(c=>((c.start_time||0)+(c.end_time||0))/2);
  const datasets = [];
  topEventNames.forEach((event,idx)=>{
    const color = EVENT_COLORS[idx % EVENT_COLORS.length];
    const probs = chunks.map(c=>{const p=(c.top_predictions||[]).find(p=>p.name===event.name);return p?p.prob:null;});
    datasets.push({label:event.name.length>50?event.name.substring(0,50)+"\u2026":event.name,data:probs.map((p,i)=>({x:times[i],y:p})),borderColor:color,backgroundColor:color,borderWidth:2,pointRadius:probs.map(p=>p!==null?getMarkerSize(p):0),pointBackgroundColor:color,pointBorderColor:"#FFF",pointBorderWidth:1,tension:.2,spanGaps:false,fill:false});
    const hpp=[];probs.forEach((p,i)=>{if(p!==null&&p>=HIGH_PROBABILITY_THRESHOLD)hpp.push({x:times[i],y:p});});
    if(hpp.length>0)datasets.push({label:null,data:hpp,backgroundColor:color+"33",borderColor:"transparent",pointRadius:hpp.map(()=>getMarkerSize(HIGH_PROBABILITY_THRESHOLD)*2),pointBackgroundColor:color+"33",pointBorderWidth:0,showLine:false,order:1});
  });
  const vt=times.filter(t=>t!=null),mn=vt.length>0?Math.min(...vt):0,mx=vt.length>0?Math.max(...vt):1;
  datasets.push({label:`Threshold (${(DEFAULT_PROBABILITY_THRESHOLD*100).toFixed(0)}%)`,data:[{x:mn,y:DEFAULT_PROBABILITY_THRESHOLD},{x:mx,y:DEFAULT_PROBABILITY_THRESHOLD}],borderColor:"#FF9800",borderDash:[5,5],borderWidth:1.5,pointRadius:0,fill:false,order:0});
  datasets.push({label:`High (${(HIGH_PROBABILITY_THRESHOLD*100).toFixed(0)}%)`,data:[{x:mn,y:HIGH_PROBABILITY_THRESHOLD},{x:mx,y:HIGH_PROBABILITY_THRESHOLD}],borderColor:"#F44336",borderDash:[2,4],borderWidth:1,pointRadius:0,fill:false,order:0});
  charts.timeline = new Chart(ctx,{type:"scatter",data:{datasets},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:"right",labels:{font:{size:9},filter:i=>i.text!==null}},tooltip:{callbacks:{label:ctx=>{if(ctx.dataset.label?.includes("Threshold")||ctx.dataset.label?.includes("High"))return null;if(ctx.raw.y===null)return"No data";return`${ctx.dataset.label}: ${(ctx.raw.y*100).toFixed(1)}%`;}}}},scales:{y:{min:-.05,max:1.05,ticks:{callback:v=>`${(v*100).toFixed(0)}%`},title:{display:true,text:"Probability"}},x:{title:{display:true,text:"Time (seconds)"}}}}});
  return canvas;
}