// JSON-lines bridge to the actual browser rules and heuristic policies.
const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');
const sourceRef=process.env.TRIO_SOURCE_REF;
if(sourceRef&&!/^[0-9a-f]{7,40}$/.test(sourceRef))throw new Error('Expected a frozen commit hash');
const html = sourceRef?require('node:child_process').execFileSync('git',['show',sourceRef+':grid-clash/index.html'],{cwd:path.join(__dirname,'..'),encoding:'utf8'}):fs.readFileSync(path.join(__dirname, '../grid-clash/index.html'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
const elements = new Map();
const document = {querySelector(selector) {
  if (!elements.has(selector)) elements.set(selector, {
    value: '2', checked: false, style: {},
    classList: {remove(){}, toggle(){}, add(){}},
    getContext: () => ({}), getBoundingClientRect: () => ({width:0}),
  });
  return elements.get(selector);
}};
let seed = 1;
const seededMath = Object.create(Math);
seededMath.random = () => {
  seed = (Math.imul(seed,1664525)+1013904223)>>>0;
  return seed/4294967296;
};
const extracted = script.slice(script.indexOf('const C='), script.indexOf('const boardbox='));
let core = extracted;
// Offline ablations only; never enabled by browser requests.
const ablation=process.env.TRIO_ABLATION||'';
if(ablation){
  if(ablation.includes('exact'))core+=`\nconst sampledReplies=atlasReplyScenarios;atlasReplyScenarios=function(player,probabilities){
    if(n!==15||a.length!==3)return sampledReplies(player,probabilities);
    let rows=[{actions:Array(3).fill(null),weight:1}];
    for(const p of a)if(p.id!==player){const dirs=canExpand(p)?legalDirs(p):[];if(!dirs.length)dirs.push(null);
      rows=rows.flatMap(row=>dirs.map(d=>{const actions=row.actions.slice();actions[p.id]=d;return {actions,weight:row.weight/dirs.length};}));}
    return rows;
  };`;
  if(ablation.includes('noprior'))core=core.replace('.5*Math.log(Math.max(probabilities[player][d],1e-8))','(n===15&&a.length===3?0:.5)*Math.log(Math.max(probabilities[player][d],1e-8))');
  if(ablation.includes('own'))core=core.replace('return atlasScoreMargin(player,projected);','return n===15&&a.length===3?projected[player]:atlasScoreMargin(player,projected);');
  if(ablation.includes('enclosure'))core+=`\nconst reactiveChoice=atlasChoose;atlasChoose=function(player,probabilities){
    if(n===15&&a.length===3){const p=a[player];
      if(!p.loopPlan?.length||p.loopPos>=p.loopPlan.length){p.loopPlan=findEnclosurePlan(p)||[];p.loopPos=0;}
      if(p.loopPlan.length){const d=p.loopPlan[p.loopPos];
        if(legalDirs(p).includes(d)&&empty(p.x+D[d][0],p.y+D[d][1])){p.loopPos++;return d;}
        p.loopPlan=[];p.loopPos=0;
      }
    }
    return reactiveChoice(player,probabilities);
  };`;
}
const game = new Function('document','Math', 'let requestedSize=30;\n'+core + `
  const originalAI=aiDirection, originalTarget=chooseNewTarget;
  const state=()=>({board:g,positions:a.map(p=>[p.x,p.y]),scores:a.map(p=>p.score),
    blocks:[...temporaryBlocks].map(k=>k.split(',').map(Number)).sort((u,v)=>u[1]-v[1]||u[0]-v[0]),
    turn,ended,idleAttempts,tailHistory,active:a.map(p=>canExpand(p))});
  return {
    reset(opts){
      requestedSize=opts.size||30;
      $('#size').value=String(requestedSize);
      $('#count').value=String(opts.count||2);init();run=true;
      if(opts.board){
        g=opts.board.map(row=>row.slice());n=g.length;
        opts.positions.forEach(([x,y],i)=>{a[i].x=x;a[i].y=y;});
        temporaryBlocks=new Set((opts.blocks||[]).map(([x,y])=>key(x,y)));
        turn=opts.turn||0;idleAttempts=opts.idleAttempts||0;tailHistory=structuredClone(opts.tailHistory||[]);recountScores();
      }
      a.forEach((p,i)=>p.strategy=(opts.styles||[])[i]||p.strategy);
      return state();
    },
    step(actions){
      aiDirection=p=>actions[p.id]??null;chooseNewTarget=()=>null;
      const advanced=resolveTurn();return {...state(),advanced};
    },
    atlas(logits,overrides={}){
      const probabilities=atlasProbabilities(logits.flat());
      // Compute ATLAS before heuristic intentions, exactly as playTurn does.
      const actions=a.map(p=>Object.hasOwn(overrides,p.id)?overrides[p.id]:(p.strategy==='atlas'?atlasChoose(p.id,probabilities):null));
      aiDirection=p=>Object.hasOwn(overrides,p.id)||p.strategy==='atlas'?actions[p.id]:(actions[p.id]=originalAI(p));
      chooseNewTarget=p=>Object.hasOwn(overrides,p.id)||p.strategy==='atlas'?null:(actions[p.id]=originalTarget(p));
      const advanced=resolveTurn();return {...state(),advanced,actions,endReason,escape:a.map(p=>p.atlasEscape)};
    },
    match3(logits,opponentLogits,seat){
      const chosen=atlasChoose(seat,atlasProbabilities(logits.flat()));
      return this.atlas(opponentLogits,{[seat]:chosen});
    },
    play(actions){
      const proposed=actions.map(()=>null);
      aiDirection=p=>proposed[p.id]=actions[p.id]==='ai'?originalAI(p):(actions[p.id]??null);
      chooseNewTarget=p=>proposed[p.id]=actions[p.id]==='ai'?originalTarget(p):null;
      const advanced=resolveTurn();return {...state(),advanced,actions:proposed};
    },
    advise(player,style){
      // Query a teacher on a private copy; neither its plans nor RNG affect play.
      const saved={g,a,temporaryBlocks,stateSeen,globalPanic,turn,occ,run,ended,aiDirection,chooseNewTarget};
      try{
        g=g.map(row=>row.slice());a=structuredClone(a);
        temporaryBlocks=new Set(temporaryBlocks);stateSeen=new Map(stateSeen);
        aiDirection=originalAI;chooseNewTarget=originalTarget;
        const p=a[player];if(style)p.strategy=style;
        const legal=legalDirs(p);
        if(!canExpand(p)||!legal.length)return {action:null};
        let action=originalAI(p);
        if(!legal.includes(action))action=originalTarget(p);
        return {action:legal.includes(action)?action:null};
      }finally{
        ({g,a,temporaryBlocks,stateSeen,globalPanic,turn,occ,run,ended,aiDirection,chooseNewTarget}=saved);
      }
    },
    capture(){resolveAllCaptures();return state();}
  };
`)(document,seededMath);
readline.createInterface({input:process.stdin}).on('line', line => {
  try {
    const msg=JSON.parse(line);
    if(msg.seed!==undefined)seed=msg.seed>>>0;
    let result;
    if(msg.op==='advise'){
      const priorSeed=seed;
      try{result=game.advise(msg.player,msg.style);}finally{seed=priorSeed;}
    }else result=msg.op==='reset'?game.reset(msg):
      msg.op==='match3'?game.match3(msg.logits,msg.opponent_logits,msg.seat):
      msg.op==='capture'?game.capture():msg.op==='atlas'?game.atlas(msg.logits,msg.overrides):msg.op==='play'?game.play(msg.actions):game.step(msg.actions);
    process.stdout.write(JSON.stringify(result)+'\n');
  } catch(error) {process.stdout.write(JSON.stringify({error:error.stack})+'\n');}
});
