const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

// Run the actual page logic with a minimal DOM; no copy of the game rules.
const html = fs.readFileSync(path.join(__dirname, '../grid-clash/index.html'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
new Function(script); // Also check the handlers excluded from this headless harness.
const elements = new Map();
const document = { querySelector(selector) {
  if (!elements.has(selector)) elements.set(selector, {
    value: selector === '#count' ? '3' : '8', checked: selector === '#human',
    style: {}, classList: { remove() {}, toggle() {}, add() {} },
    getContext: () => ({}), getBoundingClientRect: () => ({ width: 0 }),
  });
  return elements.get(selector);
} };
const core = script.slice(script.indexOf('const C='), script.indexOf('const boardbox='));
const game = new Function('document', core + `
  const originalAI=aiDirection, originalTarget=chooseNewTarget;
  return {
    setup(positions,human=true){
      $('#count').value=String(positions.length);$('#human').checked=human;init();run=true;
      g=Array.from({length:n},()=>Array(n).fill(-1));
      positions.forEach(([x,y],i)=>{a[i].x=x;a[i].y=y;g[y][x]=i;});
      recountScores();aiDirection=originalAI;chooseNewTarget=originalTarget;
    },
    choices(dirs){
      aiDirection=p=>{p.loopPos++;p.noGain++;p.visits.set('test',1);return dirs[p.id]??null;};
      chooseNewTarget=()=>null;
    },
    blocks(cells){for(const [x,y]of cells)temporaryBlocks.add(key(x,y));},
    own(x,y,id){g[y][x]=id;recountScores();},
    board(rows){n=rows.length;g=rows.map(row=>row.slice());recountScores();},
    capture:resolveAllCaptures,
    estimate(id,x,y){return estimateCapture(a[id],x,y);},
    estimatePath(id,path){return estimatePathCapture(a[id],path);},
    step:resolveTurn,
    restart:init,
    pass(id,x,y){return passable(a[id],x,y);},
    expandable(id){return canExpand(a[id]);},
    candidates(id){return frontierCandidates(a[id]);},
    idle(value){idleAttempts=value;},
    tail(collided,progress=false){return repeatedEndgameClash(collided,progress);},
    state(){return structuredClone({turn,g,a,occ,ended,idleAttempts,endReason,tailHistory,blocks:[...temporaryBlocks]});},
    hint(){return $('#joyhint').textContent;}
  };
`)(document);

game.setup([[1,2],[3,2],[8,8]]);
game.choices({1:3,2:1});
const before = game.state();
assert.equal(game.step(1), false);
let s = game.state();
assert.equal(s.turn, 0);
assert.deepEqual(s.g, before.g);
assert.deepEqual(s.a.map(p => [p.x,p.y,p.score,p.noGain,p.visits]),
  before.a.map(p => [p.x,p.y,p.score,p.noGain,p.visits]));
assert.deepEqual(s.blocks, ['2,2']);
assert.match(game.hint(), /搶格/);
for(let id=0;id<3;id++) assert.equal(game.pass(id,2,2), false);
assert.equal(game.candidates(0).some(c=>c.x===2&&c.y===2), false);

// A rejected click neither advances time nor removes the ban.
assert.equal(game.step(1), false);
assert.equal(game.state().turn, 0);
assert.deepEqual(game.state().blocks, ['2,2']);
game.choices({1:3,2:1}); // AI also attempts the forbidden square.
assert.equal(game.step(0), true);
s = game.state();
assert.equal(s.turn, 1);
assert.deepEqual([s.a[0].x,s.a[0].y], [1,1]);
assert.deepEqual([s.a[1].x,s.a[1].y], [3,2]);
assert.deepEqual([s.a[2].x,s.a[2].y], [9,8]);
assert.equal(s.g[2][2], -1);
assert.deepEqual(s.blocks, []);
assert.equal(game.pass(1,2,2), true);

// The same cancellation rule applies to AI-only collisions.
game.setup([[1,2],[3,2],[8,8]], false);
game.choices({0:1,1:3,2:0});
assert.equal(game.step(), false);
assert.equal(game.state().turn, 0);
assert.deepEqual(game.state().a.map(p=>[p.x,p.y]), [[1,2],[3,2],[8,8]]);

// Further cancellations accumulate bans until a round completes.
game.setup([[1,2],[3,2],[1,0]]);
game.choices({1:3,2:1});
assert.equal(game.step(1), false);
game.choices({1:0,2:2});
assert.equal(game.step(0), false);
assert.deepEqual(game.state().blocks.sort(), ['1,1','2,2']);
game.choices({1:0,2:1});
assert.equal(game.step(3), true);
assert.equal(game.state().turn, 1);
assert.deepEqual(game.state().blocks, []);

// A transient block does not mean permanently unable to expand.
game.setup([[0,0],[2,2]]);
game.blocks([[1,0],[0,1]]);
assert.equal(game.expandable(0), true);
game.choices({1:1});
assert.equal(game.step(1), true); // Forced wait when all human directions are blocked.
assert.deepEqual([game.state().a[0].x,game.state().a[0].y], [0,0]);
assert.equal(game.state().ended, false);
assert.deepEqual(game.state().blocks, []);
assert.match(game.hint(), /原地等待/);

game.blocks([[1,0]]);
game.restart();
assert.deepEqual(game.state().blocks, []);
assert.equal(game.state().turn, 0);

// Exercise real heuristic decisions against the movement restriction.
game.setup([[1,2],[3,2]], false);
game.blocks([[2,2]]);
game.step();
assert.equal(game.state().a.some(p=>p.x===2&&p.y===2), false);
console.log('Collision cancellation, temporary bans, expiry, AI and restart checks passed.');

// A complete wall leaves the large right-hand region exclusively reachable by A.
function divider(gaps=[]){
  game.setup([[5,15],[0,15]]);
  const board=Array.from({length:30},()=>Array(30).fill(-1));
  for(let y=0;y<30;y++)if(!gaps.includes(y))board[y][5]=0;
  board[15][0]=1;
  game.board(board);
}
divider();
assert.equal(game.capture(),720);
s=game.state();
assert.equal(s.a[0].score,750);
assert.equal(s.a[1].score,1);
assert.equal(s.g[15][0],1);
assert.equal(s.g[0][0],-1); // Both A and B can reach the left-hand blanks.
assert.equal(game.capture(),0); // One simultaneous pass is already stable.

// The same rule applies to equally sized regions.
game.setup([[2,2],[0,2]]);
const equal=Array.from({length:5},()=>[-1,-1,0,-1,-1]);
equal[2][0]=1;
game.board(equal);
assert.equal(game.capture(),10);
assert.equal(game.state().a[0].score,15);

// A temporarily blocked gap is still a permanent route; do not fill either side.
divider([10]);
game.blocks([[5,10]]);
assert.equal(game.capture(),0);
assert.equal(game.state().g[10][5],-1);
assert.deepEqual(game.state().blocks,['5,10']);

// Single-step and route estimates use exactly the same ownership rule and restore state.
divider([10]);
let saved=game.state();
assert.equal(game.estimate(0,5,10),721);
assert.deepEqual(game.state(),saved);
game.own(5,10,0);
assert.equal(game.capture(),720);
divider([28,29]);
saved=game.state();
assert.equal(game.estimatePath(0,[28*30+5,29*30+5]),722);
assert.deepEqual(game.state(),saved);
assert.equal(game.estimatePath(0,[28*30+5,15*30]),0); // Invalid route must not paint a prefix.
assert.deepEqual(game.state(),saved);

// Three players: the middle is shared, but only C reaches beyond C's wall.
game.setup([[5,15],[0,15],[20,15]]);
const three=Array.from({length:30},()=>Array(30).fill(-1));
for(let y=0;y<30;y++){three[y][5]=0;three[y][20]=2;}
three[15][0]=1;
game.board(three);
assert.equal(game.capture(),270);
assert.equal(game.state().g[10][10],-1);
assert.equal(game.state().g[10][25],2);

// Artificial disconnected territories verify that inaccessible blanks stay empty.
game.setup([[0,0],[1,0]]);
const isolated=Array.from({length:5},(_,y)=>Array.from({length:5},(_,x)=>(x+y)%2));
isolated[2][2]=-1;
game.board(isolated);
saved=game.state();
assert.equal(game.capture(),0);
assert.deepEqual(game.state(),saved);
console.log('Exclusive reachability, large/equal regions, shared/unreachable cells, temporary bans and AI estimates passed.');

// The final empty square contested simultaneously is neutral, not awarded by order.
game.setup([[0,1],[2,1]],false);
game.board([[0,0,1],[0,-1,1],[0,1,1]]);
game.choices({0:1,1:3});game.step();s=game.state();
assert.equal(s.ended,true);assert.equal(s.turn,0);
assert.deepEqual(s.a.map(p=>p.score),[4,4]);assert.equal(s.g[1][1],-1);
assert.match(s.endReason,/最後一格/);

// Two blanks: ordinary collision must NOT end the game or alter scores.
game.setup([[0,1],[2,1]],false);
game.board([[0,-1,1],[0,-1,1],[0,1,1]]);
game.choices({0:1,1:3});game.step();
assert.equal(game.state().ended,false);assert.equal(game.state().idleAttempts,1);

// Collisions count towards the no-progress safety limit, but rejected clicks do not.
game.setup([[0,1],[2,1]]);
game.board([[0,-1,1],[0,-1,1],[0,1,1]]);game.idle(17);
game.choices({1:3});game.step(3);assert.equal(game.state().idleAttempts,17);
game.step(1);assert.equal(game.state().ended,true);assert.match(game.state().endReason,/僵持/);

// Progress on the threshold attempt resets the counter instead of ending early.
game.setup([[0,1],[2,1]]);
game.board([[-1,-1,-1],[0,-1,1],[-1,-1,-1]]);game.idle(17);
game.choices({1:2});game.step(0);
assert.equal(game.state().idleAttempts,0);assert.equal(game.state().ended,false);

// A permanently trapped human can wait while the other player finishes.
game.setup([[0,0],[1,1]]);
game.board([[0,1,1],[1,1,-1],[1,1,1]]);
game.choices({1:1});game.step(1);
assert.equal(game.state().ended,true);assert.equal(game.state().g[1][2],1);
assert.deepEqual([game.state().a[0].x,game.state().a[0].y],[0,0]);
console.log('Last-square collision, non-terminal collision, no-progress safety, reset on gain, and trapped-human checks passed.');

// Alternate two contested cells: collide, move down, collide, move up.
game.setup([[0,0],[2,0]],false);
game.board([[0,-1,1],[0,-1,1],[0,0,1]]);
const cycle=[[1,3],[2,2],[1,3],[0,0]];
for(let i=0;i<12;i++){
  game.choices({0:cycle[i%4][0],1:cycle[i%4][1]});game.step();
  assert.equal(game.state().ended,i===11,'Only three COMPLETE identical cycles end play');
}
assert.match(game.state().endReason,/尾盤重複爭奪/);
assert.deepEqual(game.state().a.map(p=>p.score),[4,3]);
assert.equal(game.state().idleAttempts,12);

// Repeated positions without any collision do not trigger the new rule.
game.restart();game.setup([[0,0],[2,0]],false);
game.board([[0,-1,1],[0,-1,1],[0,0,1]]);
for(let i=0;i<20;i++)assert.equal(game.tail(false),false);
// Progress clears history, as does leaving the 2..4-cell tail-game range.
assert.equal(game.tail(true,true),false);assert.equal(game.state().tailHistory.length,0);
assert.equal(game.tail(true),false);assert.equal(game.tail(true),false);
game.blocks([[1,0]]);assert.equal(game.tail(true),false,'Different entry bans are different states');
game.board([[0,-1,1],[-1,-1,-1],[-1,0,1]]);
assert.equal(game.tail(true),false);assert.equal(game.state().tailHistory.length,0);
game.restart();assert.equal(game.state().tailHistory.length,0);
console.log('Exact three-cycle tail clash, no-collision exclusion, block identity, progress and restart resets passed.');
