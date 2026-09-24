const {chromium}=require('playwright');
const fs=require('fs');
const path=require('path');
(async()=>{
 const out=process.env.PATHGUARD_QA_DIR || 'D:/Codex_output/PathGuard数学模型与架构深化/前端检查';
 fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
 const page=await browser.newPage({viewport:{width:1440,height:1100}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.PATHGUARD_URL || 'http://127.0.0.1:8522/');
 await page.getByText('风险工作台',{exact:true}).first().click();
 await page.getByRole('tab',{name:'专业依据',exact:true}).click();
 await page.getByText('算法原理与系统架构',{exact:true}).click();
 const panel=page.locator('[data-testid="stExpander"]').filter({has:page.getByText('算法原理与系统架构',{exact:true})});
 for(const [i,name] of ['系统数据流','学习模型与规则','空间与执行公式','验证与实验协议'].entries()){
   await panel.getByRole('tab',{name,exact:true}).click();
   await panel.evaluate(e=>e.scrollIntoView({block:'start'}));
   await page.waitForTimeout(700);
   if(await panel.locator('.katex-error:visible').count())errors.push('KaTeX error: '+name);
   if(await page.locator('[data-testid="stException"]').count())errors.push('Streamlit exception: '+name);
   const images=panel.locator('img:visible');
   if(!(await images.count()))errors.push('Missing diagram: '+name);
   for(let j=0;j<await images.count();j++){
     if(!await images.nth(j).evaluate(e=>e.complete&&e.naturalWidth>0))errors.push('Broken diagram: '+name);
   }
   await page.screenshot({path:path.join(out,`技术展示-${i+1}.png`)});
 }
 if(!await panel.getByRole('button',{name:'下载完整技术项目书 Word'}).count())errors.push('Missing download');
 await page.setViewportSize({width:390,height:844});
 await panel.getByRole('tab',{name:'学习模型与规则',exact:true}).click();
 await panel.evaluate(e=>e.scrollIntoView({block:'start'}));
 await page.screenshot({path:path.join(out,'技术展示-移动端.png')});
 const horizontalOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+2);
 if(horizontalOverflow)errors.push('Page horizontal overflow');
 fs.writeFileSync(path.join(out,'result.json'),JSON.stringify({errors,horizontalOverflow},null,2));
 console.log(JSON.stringify({errors,horizontalOverflow}));
 await browser.close();if(errors.length)process.exitCode=1;
})().catch(e=>{console.error(e);process.exit(1)});
