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
    setup(state){$('#size').value=String(state.board.length);$('#count').value='2';init();
      g=structuredClone(state.board);turn=state.turn;ended=state.ended;
      a.forEach((p,i)=>{[p.x,p.y]=state.positions[i];});
      temporaryBlocks=new Set(state.blocks.map(pos=>key(...pos)));recountScores();},
    snapshot(){return JSON.stringify({g,a,turn,occ,blocks:[...temporaryBlocks],tailHistory});},
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
for(const [size,count]of [[15,3],[30,4]]){
  const result=game.size(size,count);assert.equal(result.n,size);
  assert(!result.strategies.includes('atlas'));
  assert.match(result.options,/value="atlas" disabled/);
}
console.log(fixtures.length+' Python/browser search parity fixtures passed; board sizes and ATLAS scope passed.');
