#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import re

P=Path('index_v3.html')
if not P.exists(): raise SystemExit('ERREUR: index_v3.html introuvable')
s=P.read_text(encoding='utf-8')
V='v3.77-gps-terrain'
s=re.sub(r"var APP_VER\s*=\s*'[^']+';",f"var APP_VER = '{V}';",s,count=1)
s=re.sub(r"var V\s*=\s*'[^']+';",f"var V='{V}';",s,count=1)

# Le bouton doit avoir un ID stable
s,n=re.subn(r'<button(?=[^>]*onclick="locateMe\(\)")(?=[^>]*class="fab fab-white")[^>]*>📍</button>',
'<button id="locate-btn" class="fab fab-white" title="Suivi GPS terrain" onclick="locateMe()">📍</button>',s,count=1)
if n!=1: raise SystemExit('ERREUR: bouton GPS introuvable')

CSS='''
#locate-btn.gps-active{background:#1976d2!important;color:#fff!important;box-shadow:0 0 0 4px rgba(25,118,210,.24),0 3px 10px rgba(0,0,0,.3)}
#gps-status{position:fixed;left:50%;bottom:max(10px,env(safe-area-inset-bottom));transform:translateX(-50%);z-index:850;background:rgba(26,26,46,.88);color:#fff;padding:4px 10px;border-radius:12px;font-size:10px;pointer-events:none;display:none;white-space:nowrap}
'''
if '#gps-status' not in s:s=s.replace('</style>',CSS+'</style>',1)
if '<div id="gps-status">' not in s:s=s.replace('<div id="net-toast"></div>','<div id="net-toast"></div>\n<div id="gps-status"></div>',1)

GPS=r'''// ══════════════════════════════════════════════════════════
// GPS TERRAIN v3.77 : point mis a jour a chaque position fraiche
// ══════════════════════════════════════════════════════════
function _isOnLGV(lat,lng){return lat>=44.3&&lat<=47.7&&lng>=-1.7&&lng<=1.3;}
function _restoreMapPos(){try{localStorage.removeItem('lgv_map_pos');}catch(e){}}
var GPS={watchId:null,poll:null,watchdog:null,running:false,following:true,pending:false,last:null,lastCallback:0,lastCoordChange:0,marker:null,listeners:[]};
function _gpsCircle(lng,lat,r){var a=[],R=6378137,lr=lat*Math.PI/180;for(var i=0;i<=48;i++){var q=i/48*Math.PI*2,dx=Math.cos(q)*r,dy=Math.sin(q)*r;a.push([lng+dx/(R*Math.cos(lr))*180/Math.PI,lat+dy/R*180/Math.PI]);}return{type:'Feature',properties:{},geometry:{type:'Polygon',coordinates:[a]}};}
function _gpsUi(){var b=document.getElementById('locate-btn');if(b)b.classList.toggle('gps-active',GPS.running&&GPS.following);}
function _gpsLayers(){
  if(!GPS.marker){var el=document.createElement('div');el.style.cssText='width:18px;height:18px;border-radius:50%;background:#1976d2;border:3px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.5)';GPS.marker=new maplibregl.Marker({element:el,anchor:'center'}).setLngLat([0,0]).addTo(map);}
  if(!map.getSource('_gpsAcc'))map.addSource('_gpsAcc',{type:'geojson',data:emptyFC()});
  if(!map.getLayer('_gpsAccFill'))map.addLayer({id:'_gpsAccFill',type:'fill',source:'_gpsAcc',paint:{'fill-color':'#1976d2','fill-opacity':.12}});
  if(!map.getLayer('_gpsAccLine'))map.addLayer({id:'_gpsAccLine',type:'line',source:'_gpsAcc',paint:{'line-color':'#1976d2','line-width':1,'line-opacity':.5}});
}
function _gpsStatus(p,stale){var e=document.getElementById('gps-status');if(!e)return;e.style.display='block';e.style.background=stale?'rgba(198,40,40,.92)':'rgba(26,26,46,.88)';e.textContent=(stale?'GPS ancien':'GPS actif')+' · ±'+Math.round(p.accuracy)+' m · '+new Date(p.receivedAt).toLocaleTimeString('fr-FR');}
function _gpsReceive(pos){
  GPS.pending=false;if(!pos||!pos.coords)return;
  var lat=Number(pos.coords.latitude),lng=Number(pos.coords.longitude),acc=Number(pos.coords.accuracy)||999,ts=Number(pos.timestamp)||Date.now(),now=Date.now();
  if(!isFinite(lat)||!isFinite(lng))return;
  /* Une vraie position ancienne n'ecrase jamais la derniere position. */
  if(now-ts>15000){if(GPS.last)_gpsStatus(GPS.last,true);return;}
  GPS.lastCallback=now;
  if(!GPS.last||_haversine(GPS.last.lat,GPS.last.lng,lat,lng)>.8)GPS.lastCoordChange=now;
  GPS.last={lat:lat,lng:lng,accuracy:acc,timestamp:ts,receivedAt:now};
  _gpsLayers();
  /* Aucun filtre de distance ou de precision: le point suit chaque callback frais. */
  GPS.marker.setLngLat([lng,lat]);map.getSource('_gpsAcc').setData(_gpsCircle(lng,lat,Math.max(3,Math.min(acc,500))));
  if(GPS.following)map.easeTo({center:[lng,lat],zoom:Math.max(map.getZoom(),18),duration:250,essential:true});
  _gpsStatus(GPS.last,false);_gpsUi();
  GPS.listeners.slice().forEach(function(fn){try{fn(GPS.last);}catch(e){}});
}
function _gpsError(e){GPS.pending=false;if(e&&e.code===1){stopLiveLocation();showNetToast('📍 Autorisation GPS refusee');}}
function _gpsStartWatch(){if(GPS.watchId!==null)try{navigator.geolocation.clearWatch(GPS.watchId);}catch(e){}GPS.watchId=navigator.geolocation.watchPosition(_gpsReceive,_gpsError,{enableHighAccuracy:true,maximumAge:0,timeout:12000});}
function _gpsFreshRequest(){if(!GPS.running||GPS.pending||document.visibilityState!=='visible')return;GPS.pending=true;navigator.geolocation.getCurrentPosition(_gpsReceive,function(e){GPS.pending=false;if(e&&e.code===1)_gpsError(e);},{enableHighAccuracy:true,maximumAge:0,timeout:10000});}
function startLiveLocation(follow){
  if(!navigator.geolocation){showNetToast('📍 Geolocalisation indisponible');return;}
  if(typeof follow==='boolean')GPS.following=follow;
  if(GPS.running){_gpsUi();return;}
  GPS.running=true;GPS.pending=false;GPS.last=null;GPS.lastCallback=0;GPS.lastCoordChange=Date.now();_gpsStartWatch();_gpsFreshRequest();
  clearInterval(GPS.poll);GPS.poll=setInterval(_gpsFreshRequest,5000);
  clearInterval(GPS.watchdog);GPS.watchdog=setInterval(function(){if(!GPS.running||document.visibilityState!=='visible')return;if(Date.now()-GPS.lastCallback>15000){GPS.pending=false;_gpsStartWatch();_gpsFreshRequest();showNetToast('🛰 Relance du GPS…');}},5000);
  _gpsUi();
}
function stopLiveLocation(){if(GPS.watchId!==null)try{navigator.geolocation.clearWatch(GPS.watchId);}catch(e){}clearInterval(GPS.poll);clearInterval(GPS.watchdog);GPS.watchId=null;GPS.poll=null;GPS.watchdog=null;GPS.running=false;GPS.pending=false;_gpsUi();}
function locateMe(){if(!GPS.running){GPS.following=true;startLiveLocation(true);return;}GPS.following=!GPS.following;if(GPS.following&&GPS.last)map.easeTo({center:[GPS.last.lng,GPS.last.lat],zoom:Math.max(map.getZoom(),18),duration:250});_gpsUi();showNetToast(GPS.following?'📍 Suivi actif':'📍 GPS actif, carte libre');}
window.locateMe=locateMe;window.startLiveLocation=startLiveLocation;window.stopLiveLocation=stopLiveLocation;
document.addEventListener('visibilitychange',function(){if(document.visibilityState==='visible'&&GPS.running){GPS.pending=false;GPS.lastCallback=0;_gpsStartWatch();_gpsFreshRequest();}});
map.once('load',_gpsUi);
'''
starts=['// ══════════════════════════════════════════════════════════\n// POSITION MAP + GÉOLOCALISATION','// ══════════════════════════════════════════════════════════\n// LOCALISATION TEMPS REEL','// ══════════════════════════════════════════════════════════\n// LOCALISATION GPS TEMPS REEL','// ══════════════════════════════════════════════════════════\n// GPS DIRECT']
locs=[s.find(x) for x in starts if s.find(x)>=0];start=min(locs) if locs else -1
end=s.find('// ══════════════════════════════════════════════════════════\n// DETECTION PK / ACCES / PAM LE PLUS PROCHE',start)
if start<0 or end<0:raise SystemExit('ERREUR: section GPS introuvable')
s=s[:start]+GPS+'\n'+s[end:]

FIND=r'''function findNearest(){
  if(!GPS.running)startLiveLocation(false);
  if(GPS.last&&Date.now()-GPS.last.receivedAt<10000){_finishFindNearest(GPS.last.lat,GPS.last.lng,GPS.last.accuracy);return;}
  openDetail('📍 Éléments LGV proches','<div style="padding:20px;text-align:center;color:#888">Acquisition GPS fraiche…</div>');
  var done=false,t=setTimeout(function(){if(!done){GPS.listeners=GPS.listeners.filter(function(x){return x!==got;});closeDetail();showNetToast('📍 Aucune position GPS fraiche');}},15000);
  function got(p){if(done)return;done=true;clearTimeout(t);GPS.listeners=GPS.listeners.filter(function(x){return x!==got;});_finishFindNearest(p.lat,p.lng,p.accuracy);}
  GPS.listeners.push(got);_gpsFreshRequest();
}
window.findNearest=findNearest;'''
s,n=re.subn(r'function findNearest\(\)\{.*?\n\}\nwindow\.findNearest=findNearest;',FIND,s,count=1,flags=re.S)
if n!=1:raise SystemExit('ERREUR: findNearest introuvable')
P.write_text(s,encoding='utf-8',newline='\n')
print('OK',P,V)
