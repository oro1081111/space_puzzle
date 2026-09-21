// JSON-lines bridge to the actual browser rules and heuristic policies.
const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');
const html = fs.readFileSync(path.join(__dirname, '../grid-clash/index.html'), 'utf8');
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
const core = extracted;
const game = new Function('document','Math', 'let requestedSize=30;\n'+core + `
  const originalAI=aiDirection, originalTarget=chooseNewTarget;
  const state=()=>({board:g,positions:a.map(p=>[p.x,p.y]),scores:a.map(p=>p.score),
    blocks:[...temporaryBlocks].map(k=>k.split(',').map(Number)).sort((u,v)=>u[1]-v[1]||u[0]-v[0]),
    turn,ended,active:a.map(p=>canExpand(p))});
  return {
    reset(opts){
      requestedSize=opts.size||30;
      $('#size').value=String(requestedSize);
      $('#count').value=String(opts.count||2);init();run=true;
      if(opts.board){
        g=opts.board.map(row=>row.slice());n=g.length;
        opts.positions.forEach(([x,y],i)=>{a[i].x=x;a[i].y=y;});
        temporaryBlocks=new Set((opts.blocks||[]).map(([x,y])=>key(x,y)));
        turn=opts.turn||0;recountScores();
      }
      a.forEach((p,i)=>p.strategy=(opts.styles||[])[i]||p.strategy);
      return state();
    },
    step(actions){
      aiDirection=p=>actions[p.id]??null;chooseNewTarget=()=>null;
      const advanced=resolveTurn();return {...state(),advanced};
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
      msg.op==='capture'?game.capture():msg.op==='play'?game.play(msg.actions):game.step(msg.actions);
    process.stdout.write(JSON.stringify(result)+'\n');
  } catch(error) {process.stdout.write(JSON.stringify({error:error.stack})+'\n');}
});
