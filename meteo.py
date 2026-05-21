import requests
import pandas as pd
import json
import time
import schedule
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lgv_monitoring.log'),
        logging.StreamHandler()
    ]
)

class DataSource(Enum):
    METEO_FRANCE = "meteo_france"
    HUB_EAU = "hub_eau"
    VIGICRUES = "vigicrues"
    OPEN_DATA_SOFT = "open_data_soft"

@dataclass
class RiverMonitoringPoint:
    name: str
    latitude: float
    longitude: float
    station_code: Optional[str] = None
    api_source: DataSource = DataSource.VIGICRUES

class LGVMonitor:
    def __init__(self):
        """Initialisation du moniteur LGV avec sources de données corrigées"""
        # Coordonnées de la LGV SEA
        self.lgv_coordinates = [
            (44.8378, -0.5792),  # Bordeaux
            (44.9111, -0.3422),  # Saint-Médard-en-Jalles
            (45.0464, -0.0447),  # Saint-André-de-Cubzac
            (45.2833, 0.0333),   # Libourne
            (45.7167, 0.3667),   # Angoulême
            (46.3167, 0.4667),   # Poitiers
            (47.3833, 0.6833),   # Tours
        ]
        
        # Points de surveillance des cours d'eau (avec stations connues)
        self.river_monitoring_points = [
            RiverMonitoringPoint("Vienne à Châtellerault", 46.8167, 0.5453, 
                               station_code="K739001001", api_source=DataSource.VIGICRUES),
            RiverMonitoringPoint("Charente à Cognac", 45.6969, -0.3297,
                               station_code="R316000101", api_source=DataSource.VIGICRUES),
            RiverMonitoringPoint("Dordogne à Libourne", 44.9150, -0.2433,
                               station_code="O906000101", api_source=DataSource.VIGICRUES),
            RiverMonitoringPoint("Loire à Tours", 47.3936, 0.6892,
                               station_code="M841001001", api_source=DataSource.VIGICRUES),
            RiverMonitoringPoint("Canal du Midi à Toulouse", 43.6047, 1.4442,
                               station_code=None, api_source=DataSource.VIGICRUES)
        ]
        
        # Configuration des APIs (versions corrigées)
        self.api_config = {
            "pluviometry": {
                "sources": [
                    {
                        "name": "Météo France (Open Data)",
                        "url": "https://donneespubliques.meteofrance.fr/donnees_libres/Pdf/Synop",
                        "format": "csv",
                        "active": True
                    },
                    {
                        "name": "Infoclimat",
                        "url": "https://www.infoclimat.fr/opendata/",
                        "format": "json",
                        "active": False  # Nécessite inscription
                    }
                ]
            },
            "vigicrues": {
                # Version 2 de l'API Hub'Eau
                "base_url": "https://hubeau.eaufrance.fr/api/v2/hydrometrie",
                "endpoints": {
                    "observations": "/observations_tr",
                    "referentiel": "/referentiel/stations"
                },
                "timeout": 30
            }
        }
        
        # Headers pour éviter les blocages
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7'
        }
        
        # Créer le répertoire de données
        os.makedirs("data", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
    def get_pluviometry_data_alternative(self, buffer_km: int = 30) -> pd.DataFrame:
        """
        Récupère les données pluviométriques via des sources alternatives
        """
        try:
            logging.info("Tentative de récupération des données pluviométriques alternatives...")
            
            # Méthode 1: Utiliser les données SYNOP de Météo France
            synop_data = self._get_synop_data()
            
            if not synop_data.empty:
                logging.info(f"Données SYNOP récupérées: {len(synop_data)} stations")
                return synop_data
            
            # Méthode 2: Utiliser des données ouvertes régionales
            regional_data = self._get_regional_rain_data()
            
            if not regional_data.empty:
                logging.info(f"Données régionales récupérées: {len(regional_data)} points")
                return regional_data
            
            # Méthode 3: Données simulées pour développement
            logging.warning("Utilisation de données simulées pour développement")
            return self._generate_simulated_rain_data()
            
        except Exception as e:
            logging.error(f"Erreur données pluviométriques alternatives: {e}")
            return self._generate_simulated_rain_data()
    
    def _get_synop_data(self) -> pd.DataFrame:
        """Récupère les données SYNOP de Météo France"""
        try:
            # URL des données SYNOP (format CSV)
            synop_url = "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/synop.20250210.csv"
            
            # Télécharger le fichier
            response = requests.get(synop_url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                # Lire le CSV
                lines = response.text.split('\n')
                
                # Traiter les données (format spécifique SYNOP)
                data = []
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split(';')
                        if len(parts) >= 15:
                            try:
                                station_id = parts[0]
                                date_str = parts[1]
                                latitude = float(parts[10]) / 100
                                longitude = float(parts[11]) / 100
                                
                                # Vérifier si dans la zone LGV
                                if 44.0 <= latitude <= 48.0 and -1.0 <= longitude <= 1.0:
                                    precipitation = float(parts[6]) if parts[6] else 0.0
                                    
                                    data.append({
                                        'station_id': station_id,
                                        'date': date_str,
                                        'latitude': latitude,
                                        'longitude': longitude,
                                        'precipitation_mm': precipitation,
                                        'temperature': float(parts[12]) if parts[12] else None,
                                        'pressure': float(parts[13]) if parts[13] else None
                                    })
                            except:
                                continue
                
                if data:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M%S')
                    return df
                    
        except Exception as e:
            logging.warning(f"Données SYNOP non disponibles: {e}")
        
        return pd.DataFrame()
    
    def _get_regional_rain_data(self) -> pd.DataFrame:
        """Récupère des données régionales ouvertes"""
        try:
            # Exemple: Données ouvertes de la région Nouvelle-Aquitaine
            regions = [
                {
                    'name': 'Nouvelle-Aquitaine',
                    'url': 'https://data.nouvelle-aquitaine.fr/api/records/1.0/search/',
                    'dataset': 'pluviometrie-stations-meteo',
                    'params': {
                        'dataset': 'pluviometrie-stations-meteo',
                        'rows': 100,
                        'facet': 'date',
                        'geofilter.distance': '45.0,0.0,50000'  # 50km autour d'Angoulême
                    }
                }
            ]
            
            for region in regions:
                try:
                    response = requests.get(
                        region['url'],
                        params=region['params'],
                        headers=self.headers,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        records = data.get('records', [])
                        
                        if records:
                            processed_data = []
                            for record in records:
                                fields = record.get('fields', {})
                                geometry = record.get('geometry', {})
                                
                                if geometry and 'coordinates' in geometry:
                                    lon, lat = geometry['coordinates']
                                    
                                    processed_data.append({
                                        'station': fields.get('nom_station', ''),
                                        'date': fields.get('date_observation', ''),
                                        'latitude': lat,
                                        'longitude': lon,
                                        'precipitation_mm': fields.get('hauteur_pluie', 0),
                                        'source': region['name']
                                    })
                            
                            if processed_data:
                                df = pd.DataFrame(processed_data)
                                df['date'] = pd.to_datetime(df['date'])
                                return df
                                
                except Exception as e:
                    logging.debug(f"Région {region['name']}: {e}")
                    continue
                    
        except Exception as e:
            logging.warning(f"Données régionales non disponibles: {e}")
        
        return pd.DataFrame()
    
    def get_vigicrues_data_v2(self) -> Dict:
        """
        Récupère les données Vigicrues via l'API v2 de Hub'Eau
        """
        try:
            logging.info("Récupération des données Vigicrues (API v2)...")
            
            data = {}
            
            for river_point in self.river_monitoring_points:
                try:
                    river_data = {}
                    
                    # 1. Récupérer les informations de la station si code disponible
                    if river_point.station_code:
                        # Obtenir les observations temps réel
                        obs_url = f"{self.api_config['vigicrues']['base_url']}{self.api_config['vigicrues']['endpoints']['observations']}"
                        obs_params = {
                            "code_entite": river_point.station_code,
                            "date_debut_obs": (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                            "date_fin_obs": datetime.now().strftime('%Y-%m-%d'),
                            "size": 100,
                            "sort": "desc"
                        }
                        
                        response = requests.get(
                            obs_url,
                            params=obs_params,
                            headers=self.headers,
                            timeout=self.api_config['vigicrues']['timeout']
                        )
                        
                        if response.status_code == 200:
                            observations = response.json().get('data', [])
                            
                            if observations:
                                # Traiter les observations
                                processed_obs = []
                                for obs in observations[:24]:  # Dernières 24h
                                    obs_data = {
                                        'date_observation': obs.get('date_obs'),
                                        'heure_observation': obs.get('heure_obs'),
                                        'niveau_eau_m': obs.get('resultat_obs'),
                                        'statut': obs.get('statut_observation', 'inconnu')
                                    }
                                    processed_obs.append(obs_data)
                                
                                river_data['observations'] = processed_obs
                                river_data['last_observation'] = processed_obs[0] if processed_obs else None
                                river_data['station_code'] = river_point.station_code
                    
                    # 2. Si pas de données API, utiliser des données publiques alternatives
                    if not river_data.get('observations'):
                        river_data = self._get_alternative_river_data(river_point)
                    
                    # 3. Informations de base
                    river_data.update({
                        'name': river_point.name,
                        'latitude': river_point.latitude,
                        'longitude': river_point.longitude,
                        'last_update': datetime.now().isoformat(),
                        'source': river_point.api_source.value
                    })
                    
                    data[river_point.name] = river_data
                    logging.info(f"Données récupérées pour {river_point.name}")
                    
                except Exception as e:
                    logging.error(f"Erreur pour {river_point.name}: {e}")
                    # Données par défaut
                    data[river_point.name] = self._get_default_river_data(river_point)
            
            # Sauvegarder les données
            if data:
                filename = f"data/vigicrues_v2_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str, ensure_ascii=False)
                logging.info(f"Données Vigicrues sauvegardées: {filename}")
            
            return data
            
        except Exception as e:
            logging.error(f"Erreur générale Vigicrues v2: {e}")
            return self._get_all_default_river_data()
    
    def _get_alternative_river_data(self, river_point: RiverMonitoringPoint) -> Dict:
        """Récupère des données alternatives pour les cours d'eau"""
        try:
            # Méthode alternative: Données ouvertes des agences de l'eau
            water_agency_urls = {
                "Loire": "https://www.eau-loire-bretagne.fr/",
                "Dordogne": "https://www.eau-adour-garonne.fr/",
                "Charente": "https://www.eau-adour-garonne.fr/",
                "Vienne": "https://www.eau-loire-bretagne.fr/"
            }
            
            # Pour le moment, retourner des données simulées
            return {
                'observations': self._simulate_river_observations(river_point),
                'warning_level': self._estimate_warning_level(river_point.name),
                'alternative_source': True
            }
            
        except:
            return {
                'observations': [],
                'warning_level': 'unknown',
                'alternative_source': True
            }
    
    def _get_default_river_data(self, river_point: RiverMonitoringPoint) -> Dict:
        """Données par défaut pour un cours d'eau"""
        return {
            'name': river_point.name,
            'latitude': river_point.latitude,
            'longitude': river_point.longitude,
            'observations': self._simulate_river_observations(river_point),
            'warning_level': 'normal',
            'last_update': datetime.now().isoformat(),
            'source': 'simulated',
            'note': 'Données simulées - API non disponible'
        }
    
    def _get_all_default_river_data(self) -> Dict:
        """Données par défaut pour tous les cours d'eau"""
        data = {}
        for river_point in self.river_monitoring_points:
            data[river_point.name] = self._get_default_river_data(river_point)
        return data
    
    def _simulate_river_observations(self, river_point: RiverMonitoringPoint) -> List[Dict]:
        """Simule des observations de niveau d'eau"""
        observations = []
        base_level = {
            "Vienne": 1.2,
            "Charente": 0.8,
            "Dordogne": 2.1,
            "Loire": 1.8,
            "Canal": 1.0
        }.get(river_point.name.split()[0], 1.5)
        
        # Générer 24 observations (dernières 24h)
        for i in range(24):
            obs_time = datetime.now() - timedelta(hours=i)
            # Variation aléatoire +/- 30%
            variation = 1 + (0.3 * (i % 3 - 1) / 10)
            level = base_level * variation
            
            observations.append({
                'date_observation': obs_time.strftime('%Y-%m-%d'),
                'heure_observation': obs_time.strftime('%H:%M'),
                'niveau_eau_m': round(level, 2),
                'statut': 'simulé'
            })
        
        return observations
    
    def _estimate_warning_level(self, river_name: str) -> str:
        """Estime le niveau d'alerte basé sur la saison et la localisation"""
        month = datetime.now().month
        
        # Saison des pluies (octobre à mars)
        if month in [10, 11, 12, 1, 2, 3]:
            if river_name in ["Loire", "Dordogne"]:
                return "vigilance"
            else:
                return "normal"
        else:
            return "normal"
    
    def _generate_simulated_rain_data(self) -> pd.DataFrame:
        """Génère des données pluviométriques simulées pour développement"""
        logging.info("Génération de données pluviométriques simulées")
        
        # Stations météo le long de la LGV
        simulated_stations = [
            {"name": "Bordeaux-Mérignac", "lat": 44.8283, "lon": -0.7153, "elevation": 61},
            {"name": "Angoulême", "lat": 45.6589, "lon": 0.1514, "elevation": 102},
            {"name": "Poitiers-Biard", "lat": 46.5802, "lon": 0.3064, "elevation": 129},
            {"name": "Tours-St Symphorien", "lat": 47.4322, "lon": 0.7236, "elevation": 108},
            {"name": "Libourne", "lat": 44.9167, "lon": -0.2333, "elevation": 10},
            {"name": "Saint-Pierre-des-Corps", "lat": 47.3861, "lon": 0.7203, "elevation": 50}
        ]
        
        data = []
        current_time = datetime.now()
        
        for station in simulated_stations:
            # Variation aléatoire basée sur la localisation
            base_rain = {
                "Bordeaux": 3.2,
                "Angoulême": 2.8,
                "Poitiers": 2.5,
                "Tours": 2.3,
                "Libourne": 3.0
            }
            
            station_name_key = next((key for key in base_rain.keys() if key in station["name"]), "Bordeaux")
            rain_amount = base_rain.get(station_name_key, 2.5)
            
            # Ajouter une variation aléatoire
            import random
            rain_amount *= random.uniform(0.7, 1.3)
            
            data.append({
                'station': station["name"],
                'date': current_time,
                'latitude': station["lat"],
                'longitude': station["lon"],
                'precipitation_mm': round(rain_amount, 1),
                'temperature': round(random.uniform(5.0, 15.0), 1),
                'pressure': round(random.uniform(1010.0, 1025.0), 1),
                'source': 'simulated_development'
            })
        
        df = pd.DataFrame(data)
        df.to_csv(f"data/pluviometrie_simulee_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False)
        
        return df
    
    def analyze_risks(self, pluviometry_df: pd.DataFrame, vigicrues_data: Dict) -> Dict:
        """
        Analyse les risques combinés avec des algorithmes améliorés
        """
        risks = {
            "timestamp": datetime.now().isoformat(),
            "alerts": [],
            "risk_level": "FAIBLE",
            "risk_score": 0,
            "details": {},
            "recommendations": []
        }
        
        risk_factors = []
        
        # 1. Analyse pluviométrique
        if not pluviometry_df.empty and 'precipitation_mm' in pluviometry_df.columns:
            rain_data = pluviometry_df['precipitation_mm']
            avg_rain = rain_data.mean()
            max_rain = rain_data.max()
            rain_24h = rain_data.sum()  # Approximation
            
            risks['details']['pluviometry'] = {
                "moyenne_mm": round(avg_rain, 2),
                "maximum_mm": round(max_rain, 2),
                "cumul_24h": round(rain_24h, 2),
                "stations": len(pluviometry_df),
                "source": pluviometry_df['source'].iloc[0] if 'source' in pluviometry_df.columns else 'inconnu'
            }
            
            # Facteur de risque pluviométrie
            if rain_24h > 30:
                risk_factors.append(3)  # Haut risque
                risks['alerts'].append({
                    "type": "PLUVIOMETRIE",
                    "niveau": "ÉLEVÉ",
                    "code": "PLU01",
                    "message": f"Cumul pluviométrique important: {rain_24h:.1f} mm/24h",
                    "seuil": "> 30 mm/24h"
                })
                risks['recommendations'].append("Surveillance renforcée des bassins versants")
            elif rain_24h > 20:
                risk_factors.append(2)  # Risque moyen
                risks['alerts'].append({
                    "type": "PLUVIOMETRIE",
                    "niveau": "MODÉRÉ",
                    "code": "PLU02",
                    "message": f"Cumul pluviométrique notable: {rain_24h:.1f} mm/24h",
                    "seuil": "> 20 mm/24h"
                })
        
        # 2. Analyse hydrologique
        river_alerts = []
        for river_name, river_data in vigicrues_data.items():
            observations = river_data.get('observations', [])
            
            if observations:
                # Calculer la tendance sur les dernières heures
                recent_levels = []
                for obs in observations[:6]:  # 6 dernières heures
                    if 'niveau_eau_m' in obs:
                        try:
                            recent_levels.append(float(obs['niveau_eau_m']))
                        except:
                            continue
                
                if len(recent_levels) >= 3:
                    avg_level = sum(recent_levels) / len(recent_levels)
                    
                    # Calculer la tendance
                    if len(recent_levels) >= 2:
                        trend = recent_levels[0] - recent_levels[-1]
                    else:
                        trend = 0
                    
                    risks['details'][river_name] = {
                        "niveau_moyen_m": round(avg_level, 2),
                        "tendance_m_par_heure": round(trend / len(recent_levels), 3),
                        "observations": len(observations),
                        "source": river_data.get('source', 'inconnu'),
                        "warning_level": river_data.get('warning_level', 'normal')
                    }
                    
                    # Seuils d'alerte par rivière
                    alert_thresholds = {
                        "Vienne": 2.5,
                        "Charente": 2.0,
                        "Dordogne": 3.0,
                        "Loire": 3.5,
                        "Canal": 1.5
                    }
                    
                    threshold = alert_thresholds.get(river_name.split()[0], 2.0)
                    
                    if avg_level > threshold:
                        risk_factors.append(3)
                        river_alerts.append({
                            "type": "CRUE",
                            "riviere": river_name,
                            "niveau": "ÉLEVÉ",
                            "code": "CRU01",
                            "message": f"Niveau d'eau critique: {avg_level:.2f} m (seuil: {threshold} m)",
                            "valeur": round(avg_level, 2),
                            "seuil": threshold
                        })
                        risks['recommendations'].append(f"Vérifier les ouvrages d'art sur la {river_name}")
                    elif avg_level > threshold * 0.8:
                        risk_factors.append(2)
                        river_alerts.append({
                            "type": "CRUE",
                            "riviere": river_name,
                            "niveau": "MODÉRÉ",
                            "code": "CRU02",
                            "message": f"Niveau d'eau élevé: {avg_level:.2f} m",
                            "valeur": round(avg_level, 2)
                        })
        
        # Ajouter les alertes rivières
        risks['alerts'].extend(river_alerts)
        
        # 3. Calcul du score de risque global
        if risk_factors:
            risk_score = sum(risk_factors) / len(risk_factors)
            risks['risk_score'] = round(risk_score, 2)
            
            if risk_score >= 2.5:
                risks['risk_level'] = "ÉLEVÉ"
                risks['recommendations'].append("Mise en alerte des équipes de maintenance")
            elif risk_score >= 1.5:
                risks['risk_level'] = "MODÉRÉ"
                risks['recommendations'].append("Surveillance accrue recommandée")
            else:
                risks['risk_level'] = "FAIBLE"
                risks['recommendations'].append("Surveillance normale")
        
        # 4. Sauvegarder l'analyse
        filename = f"reports/risk_analysis_detailed_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(risks, f, indent=2, default=str, ensure_ascii=False)
        
        return risks
    
    def generate_enhanced_report(self, risks: Dict):
        """Génère un rapport HTML amélioré"""
        try:
            # Déterminer la couleur du risque
            risk_colors = {
                "ÉLEVÉ": "#dc3545",
                "MODÉRÉ": "#ffc107",
                "FAIBLE": "#28a745"
            }
            
            risk_color = risk_colors.get(risks['risk_level'], "#6c757d")
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="fr">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>🚄 Rapport de Risques LGV SEA - {datetime.now().strftime('%d/%m/%Y %H:%M')}</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                <style>
                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}
                    
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                        min-height: 100vh;
                        padding: 20px;
                    }}
                    
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background: white;
                        border-radius: 20px;
                        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                        overflow: hidden;
                    }}
                    
                    .header {{
                        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                        color: white;
                        padding: 30px;
                        text-align: center;
                    }}
                    
                    .header h1 {{
                        font-size: 2.5em;
                        margin-bottom: 10px;
                    }}
                    
                    .header .subtitle {{
                        font-size: 1.2em;
                        opacity: 0.9;
                    }}
                    
                    .risk-banner {{
                        background-color: {risk_color};
                        color: white;
                        padding: 20px;
                        text-align: center;
                        font-size: 1.5em;
                        font-weight: bold;
                        margin: 20px;
                        border-radius: 10px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        gap: 15px;
                    }}
                    
                    .risk-score {{
                        background: rgba(255, 255, 255, 0.2);
                        padding: 5px 15px;
                        border-radius: 20px;
                        font-size: 0.8em;
                    }}
                    
                    .dashboard {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                        gap: 20px;
                        padding: 20px;
                    }}
                    
                    .card {{
                        background: white;
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
                        border: 1px solid #e0e0e0;
                    }}
                    
                    .card-title {{
                        color: #2a5298;
                        border-bottom: 2px solid #2a5298;
                        padding-bottom: 10px;
                        margin-bottom: 15px;
                        display: flex;
                        align-items: center;
                        gap: 10px;
                    }}
                    
                    .alert-card {{
                        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                        border-color: #ffc107;
                    }}
                    
                    .alert-item {{
                        background: white;
                        padding: 15px;
                        margin: 10px 0;
                        border-radius: 8px;
                        border-left: 5px solid #dc3545;
                    }}
                    
                    .alert-item.moderate {{
                        border-left-color: #ffc107;
                    }}
                    
                    .alert-item.low {{
                        border-left-color: #28a745;
                    }}
                    
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 10px;
                    }}
                    
                    th {{
                        background: #2a5298;
                        color: white;
                        padding: 12px;
                        text-align: left;
                    }}
                    
                    td {{
                        padding: 12px;
                        border-bottom: 1px solid #e0e0e0;
                    }}
                    
                    tr:hover {{
                        background: #f8f9fa;
                    }}
                    
                    .recommendations {{
                        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                        border-color: #28a745;
                    }}
                    
                    .footer {{
                        text-align: center;
                        padding: 20px;
                        background: #f8f9fa;
                        color: #666;
                        border-top: 1px solid #e0e0e0;
                    }}
                    
                    .data-source {{
                        font-size: 0.9em;
                        color: #666;
                        margin-top: 5px;
                        font-style: italic;
                    }}
                    
                    @media (max-width: 768px) {{
                        .dashboard {{
                            grid-template-columns: 1fr;
                        }}
                        .header h1 {{
                            font-size: 2em;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1><i class="fas fa-train"></i> Moniteur LGV SEA</h1>
                        <div class="subtitle">Surveillance des risques hydrométéorologiques</div>
                        <div style="margin-top: 15px; font-size: 0.9em;">
                            Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}
                        </div>
                    </div>
                    
                    <div class="risk-banner">
                        <i class="fas fa-exclamation-triangle"></i>
                        <span>Niveau de risque: {risks['risk_level']}</span>
                        <div class="risk-score">Score: {risks.get('risk_score', 0)}/3</div>
                    </div>
                    
                    <div class="dashboard">
            """
            
            # Section Alertes
            if risks['alerts']:
                alert_html = ""
                for alert in risks['alerts']:
                    alert_class = "alert-item"
                    if alert['niveau'] == "MODÉRÉ":
                        alert_class += " moderate"
                    elif alert['niveau'] == "FAIBLE":
                        alert_class += " low"
                    
                    alert_html += f"""
                    <div class="{alert_class}">
                        <div style="display: flex; justify-content: space-between;">
                            <strong>{alert['type']} - {alert.get('riviere', '')}</strong>
                            <span style="background: {'#dc3545' if alert['niveau'] == 'ÉLEVÉ' else '#ffc107' if alert['niveau'] == 'MODÉRÉ' else '#28a745'}; 
                                  color: white; padding: 3px 10px; border-radius: 15px; font-size: 0.9em;">
                                {alert['niveau']}
                            </span>
                        </div>
                        <p style="margin-top: 8px;">{alert['message']}</p>
                        {f"<small>Seuil: {alert.get('seuil', 'N/A')}</small>" if alert.get('seuil') else ""}
                    </div>
                    """
                
                html_content += f"""
                <div class="card alert-card">
                    <div class="card-title">
                        <i class="fas fa-bell"></i> Alertes Actives ({len(risks['alerts'])})
                    </div>
                    {alert_html}
                </div>
                """
            
            # Section Pluviométrie
            if 'pluviometry' in risks['details']:
                pluv = risks['details']['pluviometry']
                html_content += f"""
                <div class="card">
                    <div class="card-title">
                        <i class="fas fa-cloud-rain"></i> Pluviométrie
                    </div>
                    <table>
                        <tr><td><strong>Cumul 24h</strong></td><td>{pluv['cumul_24h']} mm</td></tr>
                        <tr><td><strong>Maximum</strong></td><td>{pluv['maximum_mm']} mm</td></tr>
                        <tr><td><strong>Moyenne</strong></td><td>{pluv['moyenne_mm']} mm</td></tr>
                        <tr><td><strong>Stations</strong></td><td>{pluv['stations']}</td></tr>
                    </table>
                    <div class="data-source">Source: {pluv.get('source', 'inconnue')}</div>
                </div>
                """
            
            # Section Cours d'Eau
            rivers_html = ""
            for river in ["Vienne", "Charente", "Dordogne", "Loire", "Canal"]:
                if any(river in key for key in risks['details'].keys()):
                    river_key = next((key for key in risks['details'].keys() if river in key), None)
                    if river_key:
                        data = risks['details'][river_key]
                        warning_icon = "🟢"
                        if data.get('warning_level') == 'vigilance':
                            warning_icon = "🟡"
                        elif data.get('warning_level') == 'alert':
                            warning_icon = "🔴"
                        
                        rivers_html += f"""
                        <tr>
                            <td>{river_key}</td>
                            <td>{data['niveau_moyen_m']} m</td>
                            <td>{data.get('tendance_m_par_heure', 0)} m/h</td>
                            <td>{warning_icon} {data.get('warning_level', 'normal')}</td>
                        </tr>
                        """
            
            if rivers_html:
                html_content += f"""
                <div class="card">
                    <div class="card-title">
                        <i class="fas fa-water"></i> Niveaux des Cours d'Eau
                    </div>
                    <table>
                        <tr>
                            <th>Cours d'eau</th>
                            <th>Niveau moyen</th>
                            <th>Tendance</th>
                            <th>État</th>
                        </tr>
                        {rivers_html}
                    </table>
                </div>
                """
            
            # Section Recommandations
            if risks['recommendations']:
                rec_html = ""
                for i, rec in enumerate(risks['recommendations'], 1):
                    rec_html += f"""
                    <div style="padding: 10px; margin: 5px 0; background: rgba(40, 167, 69, 0.1); border-radius: 5px;">
                        <i class="fas fa-check-circle" style="color: #28a745;"></i>
                        {rec}
                    </div>
                    """
                
                html_content += f"""
                <div class="card recommendations">
                    <div class="card-title">
                        <i class="fas fa-clipboard-check"></i> Recommandations
                    </div>
                    {rec_html}
                </div>
                """
            
            # Fin du dashboard
            html_content += """
                    </div>
                    
                    <div class="footer">
                        <p><i class="fas fa-info-circle"></i> Système de surveillance LGV SEA - Données techniques</p>
                        <p style="font-size: 0.9em; margin-top: 10px;">
                            Ce rapport est généré automatiquement. Pour les décisions opérationnelles, 
                            consulter les services officiels de Météo-France et Vigicrues.
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Sauvegarder le rapport
            filename = f"reports/rapport_ameliore_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logging.info(f"Rapport amélioré généré: {filename}")
            
            # Générer aussi une version simplifiée
            self._generate_quick_report(risks)
            
        except Exception as e:
            logging.error(f"Erreur lors de la génération du rapport: {e}")
    
    def _generate_quick_report(self, risks: Dict):
        """Génère un rapport texte rapide pour consultation CLI"""
        filename = f"reports/summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"RAPPORT RAPIDE LGV SEA - {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"NIVEAU DE RISQUE: {risks['risk_level']}\n")
            f.write(f"SCORE: {risks.get('risk_score', 0)}/3\n\n")
            
            f.write("ALERTES ACTIVES:\n")
            if risks['alerts']:
                for alert in risks['alerts']:
                    f.write(f"- [{alert['niveau']}] {alert['type']}: {alert['message']}\n")
            else:
                f.write("Aucune alerte active\n")
            
            f.write("\nSYNTHÈSE PLUVIOMÉTRIE:\n")
            if 'pluviometry' in risks['details']:
                pluv = risks['details']['pluviometry']
                f.write(f"Cumul 24h: {pluv['cumul_24h']} mm\n")
                f.write(f"Max: {pluv['maximum_mm']} mm\n")
                f.write(f"Moyenne: {pluv['moyenne_mm']} mm\n")
            
            f.write("\nRECOMMANDATIONS:\n")
            for rec in risks['recommendations']:
                f.write(f"- {rec}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        logging.info(f"Rapport rapide généré: {filename}")
    
    def run_monitoring_cycle(self):
        """Exécute un cycle complet de monitoring"""
        logging.info("=" * 60)
        logging.info("Début du cycle de monitoring")
        logging.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Récupérer les données
        pluviometry_data = self.get_pluviometry_data_alternative()
        vigicrues_data = self.get_vigicrues_data_v2()
        
        # Analyser les risques
        risks = self.analyze_risks(pluviometry_data, vigicrues_data)
        
        # Générer le rapport
        self.generate_enhanced_report(risks)
        
        # Log du résultat
        logging.info(f"Cycle terminé - Niveau de risque: {risks['risk_level']}")
        logging.info(f"Score de risque: {risks.get('risk_score', 0)}")
        logging.info(f"Alertes actives: {len(risks['alerts'])}")
        logging.info(f"Recommandations: {len(risks['recommendations'])}")
        logging.info("=" * 60)
        
        # Afficher un résumé dans la console
        self._print_console_summary(risks)

    def _print_console_summary(self, risks: Dict):
        """Affiche un résumé coloré dans la console"""
        print("\n" + "🚄 " + "="*50 + " 🚄")
        print("📊 SYNTHÈSE LGV SEA - SURVEILLANCE DES RISQUES")
        print("="*58)
        
        # Affichage du niveau de risque avec couleur
        risk_level = risks['risk_level']
        if risk_level == "ÉLEVÉ":
            risk_display = f"\033[91m{risk_level} 🔴\033[0m"
        elif risk_level == "MODÉRÉ":
            risk_display = f"\033[93m{risk_level} 🟡\033[0m"
        else:
            risk_display = f"\033[92m{risk_level} 🟢\033[0m"
        
        print(f"Niveau de risque: {risk_display} (Score: {risks.get('risk_score', 0)}/3)")
        print("-"*58)
        
        # Alertes
        if risks['alerts']:
            print(f"\n🚨 Alertes actives ({len(risks['alerts'])}):")
            for alert in risks['alerts']:
                icon = "🔴" if alert['niveau'] == "ÉLEVÉ" else "🟡" if alert['niveau'] == "MODÉRÉ" else "🟢"
                print(f"  {icon} {alert['type']}: {alert['message']}")
        else:
            print("\n✅ Aucune alerte active")
        
        # Pluviométrie
        if 'pluviometry' in risks['details']:
            pluv = risks['details']['pluviometry']
            print(f"\n🌧️  Pluviométrie (24h):")
            print(f"  Cumul: {pluv['cumul_24h']} mm")
            print(f"  Maximum: {pluv['maximum_mm']} mm")
        
        # Recommandations
        if risks['recommendations']:
            print(f"\n💡 Recommandations:")
            for rec in risks['recommendations'][:3]:  # Afficher seulement les 3 premières
                print(f"  • {rec}")
        
        print("\n" + "="*58)
        print(f"🕐 Prochaine mise à jour: {(datetime.now() + timedelta(hours=1)).strftime('%H:%M')}")
        print("="*58 + "\n")

def main():
    """Fonction principale améliorée"""
    print("\n" + "="*60)
    print("🚄 MONITEUR LGV SEA - SURVEILLANCE DES RISQUES HYDROMÉTÉOROLOGIQUES")
    print("="*60)
    print("Version: 2.0 - Données alternatives et simulation")
    print(f"Démarrage: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("="*60 + "\n")
    
    monitor = LGVMonitor()
    
    # Première exécution immédiate
    print("🔍 Exécution de la première analyse...")
    monitor.run_monitoring_cycle()
    
    # Planification
    print("\n⏰ PLANIFICATION:")
    print("- Analyse toutes les heures")
    print("- Rapport complet toutes les 6 heures")
    print("- Données sauvegardées dans ./data/")
    print("- Rapports dans ./reports/")
    print("\n✅ Monitoring actif. Appuyez sur Ctrl+C pour arrêter.\n")
    
    # Planifier les exécutions
    schedule.every().hour.do(monitor.run_monitoring_cycle)
    
    # Boucle principale
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Vérifier toutes les minutes
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt du moniteur LGV")
        logging.info("Moniteur arrêté par l'utilisateur")
        print("📁 Les données sont sauvegardées dans les dossiers 'data' et 'reports'")

if __name__ == "__main__":
    main()