// Run with Playwright available on NODE_PATH; tests only this local application.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const http=require('node:http');
const root=path.resolve(__dirname,'..');
const server=http.createServer((req,res)=>{
  const file=path.resolve(root,'.'+new URL(req.url,'http://localhost').pathname);
  if(!file.startsWith(root+path.sep)){res.writeHead(403).end();return;}
  try{
    const target=fs.statSync(file).isDirectory()?path.join(file,'index.html'):file;
    res.setHeader('Content-Type',target.endsWith('.js')?'text/javascript':target.endsWith('.html')?'text/html; charset=utf-8':'application/octet-stream');
    let content=fs.readFileSync(target);
    if(target.endsWith('index.html'))content=content.toString().replace('init();setTimeout(resize,0);',
      "window.gridTest=()=>({turn,n,run,positions:a.map(p=>[p.x,p.y]),pending:!!atlasPending});window.gridSelfPlay=async(size,count=2)=>{$('#size').value=String(size);$('#count').value=String(count);$('#human').checked=false;strategyPrefs.fill('atlas');init();run=true;let attempts=0;while(!ended&&attempts++<n*n*4)await playTurn();return {turn,ended,endReason,occ,attempts};};init();setTimeout(resize,0);");
    res.end(content);
  }catch{res.writeHead(404).end();}
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await chromium.launch({headless:true,channel:'msedge'});
  try{
    const page=await browser.newPage();
    const errors=[];page.on('pageerror',error=>errors.push(String(error)));
    await page.goto(`http://127.0.0.1:${server.address().port}/grid-clash/`);
    for(const size of [15,30]){
      await page.selectOption('#size',String(size));
      assert.equal(await page.locator('.strategy').nth(1).inputValue(),'atlas');
      const fixtures=JSON.parse(fs.readFileSync(path.join(__dirname,`atlas-fixtures${size===15?'':'-30'}.json`)));
      const timings=[];
      for(const fixture of fixtures.filter((_,i)=>i%10===0)){
        const result=await page.evaluate(async state=>{
          window.testWorker ||=new Worker('atlas-worker.js');
          const start=performance.now();
          const result=await new Promise((resolve,reject)=>{
            const timeout=setTimeout(()=>reject(new Error('Worker timeout')),60000);
            testWorker.onmessage=({data})=>{clearTimeout(timeout);resolve(data);};
            testWorker.onerror=e=>{clearTimeout(timeout);reject(new Error(e.message));};
            testWorker.postMessage({id:1,state});
          });
          return {...result,ms:performance.now()-start};
        },fixture.state);
        assert(!result.error,result.error);
        fixture.logits.flat().forEach((expected,i)=>assert(Math.abs(expected-result.logits[i])<0.0001));
        timings.push(result.ms);
      }
      await page.click('#step');
      await page.waitForFunction(()=>document.querySelector('#atlasStatus').textContent.includes('瀏覽器本機運算'));
      assert.equal((await page.evaluate(()=>gridTest())).turn,1);
      assert(!await page.locator('#atlasStatus').textContent().then(t=>t.includes('失敗')));
      console.log(`${size}x${size}: WASM policy parity passed; worker ms: ${timings.map(v=>v.toFixed(1)).join(', ')}`);
    }
    const trioFixtures=JSON.parse(fs.readFileSync(path.join(__dirname,'atlas-fixtures-3p.json')));
    const twoPlayerFixture=JSON.parse(fs.readFileSync(path.join(__dirname,'atlas-fixtures.json')))[0];
    for(const fixture of [...trioFixtures.filter((_,i)=>i%5===0),twoPlayerFixture]){
      const result=await page.evaluate(state=>new Promise((resolve,reject)=>{
        const timeout=setTimeout(()=>reject(new Error('Trio worker timeout')),60000);
        testWorker.onmessage=({data})=>{clearTimeout(timeout);resolve(data);};
        testWorker.postMessage({id:2,state});
      }),fixture.state);
      assert(!result.error,result.error);
      assert.equal(result.logits.length,fixture.state.positions.length*4);
      fixture.logits.flat().forEach((expected,i)=>assert(Math.abs(expected-result.logits[i])<0.0001));
    }
    console.log('Trio WASM parity and switching back to frozen two-player model passed.');
    const trioGame=await page.evaluate(()=>gridSelfPlay(15,3));
    assert(trioGame.ended);assert(trioGame.occ>=221);
    assert((await page.locator('#atlasStatus').textContent()).includes('Trio v1.1'));
    console.log('Trio full self-play: '+JSON.stringify(trioGame));
    for(const size of ['15','30'])for(let count=3;count<=8;count++){
      await page.selectOption('#size',size);await page.selectOption('#count',String(count));
      assert.equal(await page.locator('option[value="atlas"]:disabled').count(),0);
      for(const selector of await page.locator('.strategy').all())await selector.selectOption('atlas');
      const start=Date.now();await page.click('#step');
      await page.waitForFunction(()=>document.querySelector('#atlasStatus').textContent.includes('瀏覽器本機運算'));
      assert.equal((await page.evaluate(()=>gridTest())).turn,1);
      console.log('Multiplayer '+size+' / '+count+' first turn ms: '+(Date.now()-start));
    }
    await page.selectOption('#count','3');
    await page.selectOption('#size','15');await page.check('#human');
    await page.locator('.strategy').nth(0).selectOption('apex');
    await page.locator('.strategy').nth(1).selectOption('atlas');
    await page.click('#start');
    const mixedCanvas=await page.locator('#c').boundingBox();
    await page.locator('#c').click({position:{x:mixedCanvas.width/2,y:mixedCanvas.height*.9}});
    await page.waitForFunction(()=>document.querySelector('#atlasStatus').textContent.includes('瀏覽器本機運算'));
    assert.equal((await page.evaluate(()=>gridTest())).turn,1);
    await page.uncheck('#human');
    await page.selectOption('#count','2');
    await page.locator('.strategy').nth(1).selectOption('atlas');
    await page.selectOption('#size','15');
    await page.check('#human');
    await page.click('#start');
    const canvas=await page.locator('#c').boundingBox();
    await page.locator('#c').click({position:{x:canvas.width/2,y:canvas.height*.9}});
    await page.waitForFunction(()=>document.querySelector('#atlasStatus').textContent.includes('瀏覽器本機運算'));
    assert.equal((await page.evaluate(()=>gridTest())).turn,1);
    await page.click('#start'); // Discard pending results when changing game.
    await page.selectOption('#size','30');
    await page.screenshot({path:path.join(root,'training/runs/atlas-browser.png'),fullPage:true});
    for(const size of [15,30]){
      const result=await page.evaluate(size=>gridSelfPlay(size),size);
      assert(result.ended);assert.equal(result.occ,size*size);
      console.log('Full browser ATLAS self-play '+size+': '+JSON.stringify(result));
    }
    for(const size of (process.env.ATLAS_SMOKE?[]:[15,30]))for(const count of [3,4]){
      const result=await page.evaluate(([size,count])=>gridSelfPlay(size,count),[size,count]);
      assert(result.ended);assert(result.occ>=size*size-4);
      console.log('Full multiplayer '+size+' / '+count+': '+JSON.stringify(result));
    }
    await page.selectOption('#size','30');
    await page.selectOption('#count','2');
    await page.uncheck('#human');
    await page.evaluate(()=>{
      window.Worker=class {
        constructor(){window.delayedWorker=this;}
        postMessage(data){this.request=data;}
        terminate(){}
      };
    });
    await page.click('#step');
    assert((await page.evaluate(()=>gridTest())).pending);
    await page.selectOption('#size','15');
    await page.evaluate(()=>delayedWorker.onmessage({data:{id:delayedWorker.request.id,logits:Array(8).fill(0)}}));
    let state=await page.evaluate(()=>gridTest());
    assert.equal(state.turn,0);assert.equal(state.n,15);assert.equal(state.run,false);
    await page.click('#step');
    await page.click('#pause');
    await page.evaluate(()=>delayedWorker.onmessage({data:{id:delayedWorker.request.id,logits:Array(8).fill(0)}}));
    state=await page.evaluate(()=>gridTest());assert.equal(state.turn,0);assert.equal(state.run,false);
    await page.click('#step');
    await page.evaluate(()=>delayedWorker.onmessage({data:{id:delayedWorker.request.id,error:'test offline'}}));
    await page.waitForFunction(()=>document.querySelector('#atlasStatus').textContent.includes('失敗'));
    state=await page.evaluate(()=>gridTest());assert.equal(state.turn,0);assert.equal(state.run,false);
    assert.deepEqual(errors,[]);
    console.log('Human turn, size switch, scope UI, stale-result cancellation, pause and offline failure passed.');
  }finally{await browser.close();server.close();}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});
