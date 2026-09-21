const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const root=path.join(__dirname,'..');
const html=fs.readFileSync(path.join(root,'grid-clash/index.html'),'utf8');
const script=html.split('<script>')[1].split('</script>')[0];
const elements=new Map();
const document={querySelector(selector){
  if(!elements.has(selector))elements.set(selector,{value:selector==='#size'?'15':'2',checked:false,
    style:{},classList:{remove(){},toggle(){},add(){}},getContext:()=>({}),getBoundingClientRect:()=>({width:0})});
  return elements.get(selector);
}};
const core=script.slice(script.indexOf('const C='),script.indexOf('const boardbox='));
const game=new Function('document',core+`
  return {
    setup(state){$('#size').value=String(state.board.length);$('#count').value=String(state.positions.length);init();
      g=structuredClone(state.board);n=g.length;turn=state.turn;ended=state.ended;
      a.forEach((p,i)=>{[p.x,p.y]=state.positions[i];});
      temporaryBlocks=new Set(state.blocks.map(pos=>key(...pos)));recountScores();},
    snapshot(){return JSON.stringify({g,a,turn,occ,blocks:[...temporaryBlocks],tailHistory});},
    race:atlasRaceValue,branch:atlasBranchValue,
    replies(player){return atlasReplyScenarios(player,atlasProbabilities(Array(a.length*4).fill(0)));},
    legal(){return a.map(p=>canExpand(p)?legalDirs(p):[null]);},
    plannedStep(block){
      const p=a[0],d=legalDirs(p)[0];p.loopPlan=[d];p.loopPos=0;
      if(block)temporaryBlocks.add(key(p.x+D[d][0],p.y+D[d][1]));
      const before=JSON.stringify({g,positions:a.map(q=>[q.x,q.y]),turn,occ});
      const selected=atlasTrioPlan(p);
      return {selected,d,plan:p.loopPlan,pos:p.loopPos,unchanged:before===JSON.stringify({g,positions:a.map(q=>[q.x,q.y]),turn,occ})};
    },
    actions(logits){const probs=atlasProbabilities(logits.flat());return a.map(p=>atlasChoose(p.id,probs));},
    size(size,count){$('#size').value=String(size);$('#count').value=String(count);init();return {n,strategies:a.map(p=>p.strategy),options:$('#list').innerHTML};}
  };
`)(document);
const fixtures=['atlas-fixtures.json','atlas-fixtures-30.json'].flatMap(file=>JSON.parse(fs.readFileSync(path.join(__dirname,file))));
for(const fixture of fixtures){
  game.setup(fixture.state);const before=game.snapshot();
  assert.deepEqual(game.actions(fixture.logits),fixture.actions);
  assert.equal(game.snapshot(),before,'Search must not change the actual game');
}
assert.equal(game.size(15,2).strategies[1],'atlas');
assert.equal(game.size(30,2).strategies[1],'atlas');
for(const size of [15,30])for(let count=2;count<=8;count++){
  const result=game.size(size,count);assert.equal(result.n,size);
  assert(result.strategies.includes('atlas'));
  assert(!result.options.includes(' disabled'));
}
const positions=[[0,0],[4,4],[1,2],[3,2]],board=Array.from({length:5},()=>Array(5).fill(-1));
positions.forEach(([x,y],i)=>board[y][x]=i);
game.setup({board,positions,turn:0,ended:false,blocks:[]});
const snapshot=game.snapshot();
assert.equal(game.branch(0,[1,null,1,3]),game.race(0),'Collision of players 2 and 3 cancels ALL moves');
assert.equal(game.snapshot(),snapshot);
console.log(fixtures.length+' Python/browser search parity fixtures passed; board sizes and ATLAS scope passed.');
game.size(15,3);
for(let player=0;player<3;player++){
  const replies=game.replies(player),legal=game.legal();
  assert.equal(replies.length,legal.filter((_,i)=>i!==player).reduce((size,dirs)=>size*dirs.length,1));
  assert.equal(new Set(replies.map(r=>JSON.stringify(r.actions))).size,replies.length);
  assert(Math.abs(replies.reduce((sum,r)=>sum+r.weight,0)-1)<1e-10);
  for(const row of replies)row.actions.forEach((d,i)=>assert(i===player?d===null:legal[i].includes(d)));
}
let plan=game.plannedStep(false);assert.equal(plan.selected,plan.d);assert.equal(plan.pos,1);assert(plan.unchanged);
game.size(15,3);plan=game.plannedStep(true);assert.equal(plan.selected,null);assert.deepEqual(plan.plan,[]);assert(plan.unchanged);
console.log('Three-player exhaustive replies and legal/blocked multi-step plan checks passed.');
