import test from 'node:test';
import assert from 'node:assert/strict';
import { inventoryChoice } from '../src/setup-provider-inventory.js';
import type { CoreClient } from '../src/core.js';
import type { Prompter } from '../src/prompts.js';
import type { Connection } from '../src/types.js';

for (const existing of [true, false]) test(existing ? 'local Hermes authentication is reused without key entry' : 'masked key entry saves a reference and retries discovery', async () => {
 const connection: Connection = {provider:'hermes',label:'Hermes',config:{api_key_env:'API_SERVER_KEY'},deployment:{kind:'api'}};
 const notes: string[] = []; const drafts: unknown[] = []; let entered = false;
 const core = {call: async (action: string, payload: any) => {
  if (action === 'connection.discover') return connection.config.credential_source
   ? {state:'ready',items:[{id:'model',label:'model'}],next_cursor:null}
   : {state:'needs-auth',items:[],next_cursor:null,detail:'missing'};
  if (action === 'connection.auth.prepare') return existing ? {state:'ready',config:{...connection.config,credential_source:'hermes',credential_ref:'/test/.env'}} : {state:'needs-auth'};
  assert.equal(action,'connection.auth.save'); assert.equal(payload.key,'test-secret');
  return {config:{...connection.config,credential_source:'keyring',credential_ref:'reference'}};
 }} as unknown as CoreClient;
 const ui = {password: async () => {assert.equal(existing,false); entered=true; return 'test-secret';},note:(value:string)=>notes.push(value),
  select:async()=>{throw new Error('authentication must not require manually retrying inventory');}} as unknown as Prompter;
 const model = await inventoryChoice(core,ui,connection,'llm','models','ko','',false,async()=>{},async()=>{drafts.push(structuredClone(connection));});
 assert.equal(model,'model'); assert.equal(entered,!existing); assert.equal(drafts.length,1);
 assert.ok(!JSON.stringify({drafts,notes}).includes('test-secret'));
});
