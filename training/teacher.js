// Offline only: inserted inside the actual browser engine's lexical scope.
let candidateWeights={};
const teacherOriginalWeights=styleWeights;
styleWeights=style=>style==='candidate'?{...teacherOriginalWeights('apex'),...candidateWeights}:teacherOriginalWeights(style);
const teacherOriginalPhase=adaptivePhase;
adaptivePhase=p=>{
  const phase=teacherOriginalPhase(p);
  if(p.strategy==='candidate'&&candidateWeights.scaledPhase){
    phase.escape=phase.selfFreedom<150*n*n/900;
    phase.pressure=phase.enemyFreedom<170*n*n/900;
  }
  return phase;
};
const teacherLegacyAI=originalAI;
originalAI=p=>{
  if(p.strategy==='candidate'&&candidateWeights.interiorFallback&&turn>=8&&
      !g[0].includes(p.id)&&!g[n-1].includes(p.id)&&!g.some(row=>row[0]===p.id||row[n-1]===p.id)){
    const style=p.strategy;p.strategy=candidateWeights.interiorFallback==='reactive'?'atlas':candidateWeights.interiorFallback;
    try{
      if(candidateWeights.interiorFallback==='reactive')return atlasChoose(p.id,a.map(q=>{
        const legal=legalDirs(q);return D.map((_,d)=>legal.includes(d)?1/Math.max(1,legal.length):0);
      }));
      return teacherLegacyAI(p);
    }finally{p.strategy=style;}
  }
  if(p.strategy!=='candidate'||!candidateWeights.avoidClash)return teacherLegacyAI(p);
  const prior=temporaryBlocks,blocked=new Set(prior);
  for(const q of a)if(q.id!==p.id&&canExpand(q))for(const d of legalDirs(q)){
    const x=q.x+D[d][0],y=q.y+D[d][1];
    if(g[y][x]===-1)blocked.add(key(x,y));
  }
  temporaryBlocks=blocked;
  if(!legalDirs(p).length)temporaryBlocks=prior;
  try{return teacherLegacyAI(p);}finally{temporaryBlocks=prior;}
};
// Preserve the legacy planner; transformations apply only to candidate policies.
// This is a local source transformation, not evaluation of external input.
const teacherOriginalPlan=findEnclosurePlan;
const raceNeedle='if((meaningfulCapture||meaningfulTrap||meaningfulEdge)&&score>bestScore)';
for(const needle of [raceNeedle,'const trap=confinementGainWithCells(p,path,trapBase);',
  'bbox*1.75+','turnBonus+','for(let d=0;d<4;d++){','Math.random()*.05;']){
  if(!teacherOriginalPlan.toString().includes(needle))throw new Error('Planner source changed; review candidate transformation: '+needle);
}
const teacherRacePlan=eval('('+teacherOriginalPlan.toString().replace(raceNeedle,
  'score-=teacherRaceCost(p,path)*(candidateWeights.raceCost||0);if(meaningfulCapture||meaningfulTrap||meaningfulEdge)teacherOfferPlan(score,dirs);'+raceNeedle)
  .replace('const trap=confinementGainWithCells(p,path,trapBase);',
    'const trap=confinementGainWithCells(p,path,trapBase);if(candidateWeights.topology)teacherCorrectTrap(trap,path,trapBase);')
  .replace('bbox*1.75+', 'bbox*(candidateWeights.bbox??1.75)+')
  .replace('turnBonus+', 'turnBonus*(candidateWeights.turnBias??1)+')
  .replace('for(let d=0;d<4;d++){','for(const d of teacherDirections(p)){')
  .replace('Math.random()*.05;', 'Math.random()*.05-(candidateWeights.beamRace?candidateWeights.beamRace*teacherRaceCost(p,path):0);')+')');
let teacherArrival=null;
findEnclosurePlan=p=>{
  if(p.strategy!=='candidate')return teacherOriginalPlan(p);
  const previous=teacherArrival;
  teacherArrival=new Float64Array(n*n).fill(Infinity);
  for(const q of a)if(q.id!==p.id&&q.active)for(let y=0;y<n;y++)for(let x=0;x<n;x++){
    const z=y*n+x;teacherArrival[z]=Math.min(teacherArrival[z],Math.abs(q.x-x)+Math.abs(q.y-y));
  }
  try{return candidateWeights.cuts?teacherCuts(p):teacherRacePlan(p);}
  finally{teacherArrival=previous;}
};
function teacherDirections(p){
  const mode=candidateWeights.directionBias||0,dirs=[0,1,2,3];
  if(mode===1)return dirs.reverse();
  if(mode>=2){
    const distance=d=>Math.abs(p.x+D[d][0]-(n-1)/2)+Math.abs(p.y+D[d][1]-(n-1)/2);
    dirs.sort((u,v)=>(mode===2?1:-1)*(distance(u)-distance(v)));
  }
  return dirs;
}
function teacherCorrectTrap(trap,path,base){
  // Remove ordinary trail occupancy and pre-existing confinement from cut reward.
  trap.gain=Math.max(0,trap.gain-path.length*base.length);
  trap.tight=Math.max(0,trap.tight-base.reduce((s,b)=>s+Math.max(0,100-b.freedom),0));
  trap.locked=Math.max(0,trap.locked-base.filter(b=>b.freedom<=45).length);
}
let teacherCutPool=null;
function teacherOfferPlan(value,dirs){
  if(!teacherCutPool||value<=0)return;
  if(teacherCutPool.length>=32&&value<=teacherCutPool[teacherCutPool.length-1].value)return;
  teacherCutPool.push({value,dirs:dirs.slice()});teacherCutPool.sort((u,v)=>v.value-u.value);
  if(teacherCutPool.length>32)teacherCutPool.pop();
}
function teacherCuts(p){
  const W=styleWeights('candidate'),base=confinementBaseline(p);
  let best=null,bestScore=0;
  const seen=new Set();
  function score(dirs){
    const signature=dirs.join('');if(seen.has(signature))return;seen.add(signature);
    let x=p.x,y=p.y;const path=[];
    for(const d of dirs){x+=D[d][0];y+=D[d][1];if(!empty(x,y))return;path.push(y*n+x);}
    if(!path.length)return;
    const gain=estimatePathCapture(p,path),extra=gain-path.length;
    const trap=confinementGainWithCells(p,path,base);
    if(candidateWeights.topology)teacherCorrectTrap(trap,path,base);
    if(extra<2&&trap.gain<18&&trap.locked===0)return;
    const value=(extra*W.loopArea+extra/path.length*W.loopEff+trap.gain*W.trap+
      trap.tight*W.trapTight+trap.locked*W.trapLock+pathEdgeValue(path)*W.edgePlan)/
      Math.pow(path.length,candidateWeights.lengthPower??1)-teacherRaceCost(p,path)*(W.raceCost||0)-path.length*W.loopLen;
    teacherOfferPlan(value,dirs);
    if(value>bestScore){bestScore=value;best=dirs;}
  }
  // Axis-aligned cuts, at most one bend, reaching an edge or existing territory.
  for(const d of teacherDirections(p))for(let length=1;length<n;length++){
    const x=p.x+D[d][0]*length,y=p.y+D[d][1]*length;
    if(!empty(x,y))break;
    const first=Array(length).fill(d);score(first);
    for(const bend of [(d+1)%4,(d+3)%4]){
      let nx=x,ny=y;const dirs=first.slice();
      while(dirs.length<2*n){
        nx+=D[bend][0];ny+=D[bend][1];if(!empty(nx,ny))break;
        dirs.push(bend);
        if(nx===0||ny===0||nx===n-1||ny===n-1||ownAdj(p,nx,ny)>0)score(dirs.slice());
      }
    }
  }
  return best;
}
function teacherRaceCost(p,path){
  let cost=0;
  for(let i=0;i<path.length;i++){
    cost+=Math.max(0,i+1-teacherArrival[path[i]]);
  }
  return cost;
}
function teacherSnapshot(){
  return {g,a,temporaryBlocks,stateSeen,globalPanic,turn,occ,run,ended,
    idleAttempts,endReason,tailHistory,aiDirection,chooseNewTarget};
}
function teacherRestore(s){
  ({g,a,temporaryBlocks,stateSeen,globalPanic,turn,occ,run,ended,
    idleAttempts,endReason,tailHistory,aiDirection,chooseNewTarget}=s);
}
function teacherCopy(s){
  teacherRestore({...s,g:s.g.map(r=>r.slice()),a:structuredClone(s.a),
    temporaryBlocks:new Set(s.temporaryBlocks),stateSeen:new Map(s.stateSeen),
    tailHistory:structuredClone(s.tailHistory)});
  // Real turns replace these callbacks with fixed actions. Planning must never
  // inherit that previous-turn callback (in particular its null own fallback).
  aiDirection=originalAI;chooseNewTarget=originalTarget;
}
function teacherNeutral(p){
  // No opponent private routes, visits, cooldowns, styles or intentions.
  return {...p,strategy:'apex',commit:null,enclosurePlan:null,planBlocked:0,
    noGain:0,panic:0,visits:new Map([[key(p.x,p.y),1]]),recent:[],blocked:new Map(),
    loopPlan:[],loopPos:0,loopCooldown:0,atlasEscape:false,atlasTarget:null,atlasAvoid:null};
}
function teacherProposal(p){
  const legal=legalDirs(p);
  let action=originalAI(p);
  // Match resolveTurn's fallback after a planner temporarily excludes risky cells.
  if(!legal.includes(action)){
    p.commit=null;p.loopPlan=[];p.loopPos=0;
    action=originalTarget(p);
  }
  return legal.includes(action)?action:null;
}
function teacherChoose(player,opts={}){
  const horizon=opts.horizon??16, profiles=opts.profiles??3;
  const saved=teacherSnapshot(),random=Math.random;
  let rng=1;
  Math.random=()=>{rng=(Math.imul(rng,1664525)+1013904223)>>>0;return rng/4294967296;};
  const styles=['apex','bastion','sweep'],rows=[];
  try{
    if(!canExpand(a[player])||!legalDirs(a[player]).length)return {action:null,rows};
    // Keep a viable route instead of re-selecting an incompatible plan every turn.
    const p=a[player];
    if(!opts.replan&&p.loopPlan?.length>p.loopPos&&empty(p.x+D[p.loopPlan[p.loopPos]][0],p.y+D[p.loopPlan[p.loopPos]][1])){
      teacherCopy(saved);const action=teacherProposal(a[player]),memory=structuredClone(a[player]);
      teacherRestore(saved);a[player]=memory;return {action,rows,retained:true};
    }
    const candidates=[];
    for(const style of opts.topCuts&&!opts.mixedPlans?['candidate']:opts.candidate?['candidate',...styles]:styles){
      teacherCopy(saved);rng=opts.seed??101;
      a[player].strategy=style;a[player].loopPlan=[];a[player].loopPos=0;a[player].loopCooldown=0;a[player].commit=null;
      let action,pool=[];
      try{teacherCutPool=opts.topCuts?pool:null;action=teacherProposal(a[player]);}
      finally{teacherCutPool=null;}
      if(!legalDirs(a[player]).includes(action))continue;
      const memory=structuredClone(a[player]);
      const signature=JSON.stringify([action,memory.loopPlan,memory.commit?.x,memory.commit?.y]);
      if(!candidates.some(c=>c.signature===signature))candidates.push({action,memory,signature,style});
      for(const option of pool.sort((u,v)=>v.value-u.value).slice(0,opts.topCuts||0)){
        const memory=structuredClone(a[player]);memory.loopPlan=option.dirs;memory.loopPos=1;memory.commit=null;
        const signature=JSON.stringify([option.dirs[0],option.dirs]);
        if(!candidates.some(c=>JSON.stringify([c.action,c.memory.loopPlan])===signature))
          candidates.push({action:option.dirs[0],memory,signature,style:'cut'});
      }
    }
    if(opts.directions){
      teacherCopy(saved);
      for(const action of legalDirs(a[player])){
        const memory=structuredClone(saved.a[player]);
        memory.strategy=opts.candidate?'candidate':'apex';memory.commit=null;
        memory.loopPlan=[];memory.loopPos=0;memory.loopCooldown=0;
        candidates.push({action,memory,signature:'direction'+action,style:'direction'+action});
      }
    }
    const enemies=saved.a.filter(p=>p.id!==player).map(p=>p.id);
    if(candidates.length===1){
      teacherRestore(saved);a[player]=candidates[0].memory;
      return {action:candidates[0].action,rows,single:true};
    }
    let incumbent=-Infinity;
    for(const c of candidates){
      const values=[];let simulated=0,pruned=false,upperBound=null;
      for(let sample=0;sample<profiles;sample++){
        teacherCopy(saved);rng=((opts.seed??101)+sample*104729)>>>0;
        a=a.map(p=>p.id===player?structuredClone(c.memory):teacherNeutral(p));
        enemies.forEach((id,j)=>a[id].strategy=styles[profiles===9?(j===0?Math.floor(sample/3):sample%3):sample%3]);
        let attempts=0,advanced=0;
        aiDirection=p=>{
          // Align each actor's random stream at each attempt across candidates.
          // Own planning must not shift the opponent's later tie-breaking draws.
          rng=((opts.seed??101)+sample*104729+attempts*1009+p.id*65537)>>>0;
          return attempts===0&&p.id===player?c.action:originalAI(p);
        };
        chooseNewTarget=originalTarget;
        while(!ended&&advanced<horizon&&attempts<Math.max(4,3*horizon)){
          if(resolveTurn())advanced++;attempts++;
        }
        simulated+=attempts;
        const scores=a.map(p=>p.score),best=Math.max(...scores),win=scores[player]===best?1/scores.filter(x=>x===best).length:0;
        // Exact terminal outcomes; nonterminal race estimate is an explicit approximation.
        const value=ended?win+(scores[player]-Math.max(...scores.filter((_,i)=>i!==player)))/(n*n)*.1:
          .5+atlasRaceValue(player)/(n*n)*.5;
        values.push(value);
        // Every scenario is <= 1.1 (win share plus normalized margin).
        // Skip only candidates that cannot beat an already fully scored route.
        upperBound=(values.reduce((x,y)=>x+y,0)+(profiles-values.length)*1.1)/profiles;
        if(opts.prune&&values.length<profiles&&upperBound<incumbent-1e-9){pruned=true;break;}
      }
      const score=pruned?upperBound:values.reduce((x,y)=>x+y,0)/values.length;
      if(!pruned)incumbent=Math.max(incumbent,score);
      rows.push({style:c.style,action:c.action,plan:c.memory.loopPlan.slice(),
        score,values,simulated,pruned}); // Pruned scores are bounds, not training labels.
    }
    let best=0;rows.forEach((r,i)=>{if(r.score>rows[best].score+1e-10)best=i;});
    teacherRestore(saved);if(!candidates.length)return {action:null,rows};
    a[player]=candidates[best].memory;
    return {action:candidates[best].action,rows};
  }catch(e){teacherRestore(saved);throw e;}
  finally{Math.random=random;}
}
