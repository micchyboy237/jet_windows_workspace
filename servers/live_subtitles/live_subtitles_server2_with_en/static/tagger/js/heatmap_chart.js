// ===== Chunk Heatmap =====
function createChunkHeatmap(chunks, topEventNames) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const container = document.createElement("div");
  container.className = "plot-container xlarge";
  const nEvents = topEventNames.length, nChunks = chunks.length;
  const matrix = buildHeatmapData(chunks, topEventNames.map(e=>e.name));
  const datasets = [];
  for (let r=0; r<nEvents; r++) for (let c=0; c<nChunks; c++) datasets.push({x:c, y:r, v:matrix[r][c]});
  charts.heatmap = new Chart(ctx, {
    type:"matrix",
    data:{datasets:[{label:"Probability",data:datasets,backgroundColor(ctx){return getHeatmapColor(ctx.dataset.data[ctx.dataIndex].v);},borderColor:"#DDD",borderWidth:1,width:({chart})=>(chart.chartArea||{}).width/Math.max(nChunks,1)-1,height:({chart})=>(chart.chartArea||{}).height/Math.max(nEvents,1)-1}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{title:(items)=>{const ci=items[0].dataset.data[items[0].dataIndex].x,ch=chunks[ci];return`Chunk ${ch.chunk_index||ci}: ${(ch.start_time||0).toFixed(1)}s-${(ch.end_time||0).toFixed(1)}s`;},label:(item)=>{const ei=item.dataset.data[item.dataIndex].y;return`${topEventNames[ei].name}: ${(item.dataset.data[item.dataIndex].v*100).toFixed(1)}%`;}}}},scales:{x:{type:"linear",offset:true,ticks:{stepSize:1,callback:(v)=>{const i=Math.round(v);if(i>=0&&i<nChunks)return`${(chunks[i].start_time||0).toFixed(1)}s`;return"";},font:{size:8},maxRotation:60},title:{display:true,text:"Chunk Start Time"},grid:{display:false}},y:{type:"linear",offset:true,ticks:{stepSize:1,callback:(v)=>{const i=Math.round(v);if(i>=0&&i<nEvents){const n=topEventNames[i].name;return n.length>40?n.substring(0,40)+"\u2026":n;}return"";},font:{size:9}},title:{display:true,text:"Event Label"},grid:{display:false},reverse:true}}},
    plugins:[{id:"hl",afterDraw(chart){const{ctx,scales:{x,y},data}=chart,ds=data.datasets[0];ctx.save();ctx.font="9px sans-serif";ctx.textAlign="center";ctx.textBaseline="middle";ds.data.forEach(p=>{const xp=x.getPixelForValue(p.x),yp=y.getPixelForValue(p.y),bw=x.getPixelForValue(p.x+.5)-x.getPixelForValue(p.x-.5),bh=Math.abs(y.getPixelForValue(p.y+.5)-y.getPixelForValue(p.y-.5));if(p.v>0&&bw>20&&bh>15){ctx.fillStyle=getTextColor(p.v);if(p.v>=HIGH_PROBABILITY_THRESHOLD)ctx.font="bold 11px sans-serif";else if(p.v>=MEDIUM_PROBABILITY_THRESHOLD)ctx.font="bold 10px sans-serif";ctx.fillText((p.v*100).toFixed(0)+"%",xp,yp);if(p.v>=HIGH_PROBABILITY_THRESHOLD){ctx.strokeStyle="#FFD700";ctx.lineWidth=2;ctx.strokeRect(xp-bw/2,yp-bh/2,bw,bh);}}else if(p.v===0&&bw>15&&bh>10){ctx.fillStyle="#CCC";ctx.font="8px sans-serif";ctx.fillText("\u2014",xp,yp);}});ctx.restore();}}]
  });
  container.appendChild(canvas);
  const legend=document.createElement("div");legend.className="heatmap-legend";legend.innerHTML='<span>0%</span><div class="heatmap-gradient"></div><span>100%</span>';container.appendChild(legend);
  return container;
}