"""
Coverage Area Maximization Framework
=====================================
A comprehensive system for evaluating offensive player positioning against
defensive coverage, with specific focus on zone coverage responsibilities
and synergistic route combinations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist
from .zone_coverage import ZoneCoverage
import warnings
warnings.filterwarnings('ignore')

class CoverageAreaAnalyzer:
    """Main analyzer for coverage area maximization"""
    
    def __init__(self, play_data: pd.DataFrame):
        """
        Initialize with play tracking data
        
        Args:
            play_data: DataFrame with columns [x, y, player_name, player_position, 
                      player_side, frame_id, etc.]
        """
        self.data = play_data
        self.field_width = 53.3
        self.field_length = 120
        self.los = play_data['absolute_yardline_number'].iloc[0]
        
        # Separate offense and defense
        self.offense_data = play_data[play_data['player_side'] == 'Offense']
        self.defense_data = play_data[play_data['player_side'] == 'Defense']
        
        # Get unique players
        self.offensive_players = self.offense_data['nfl_id'].unique()
        self.defensive_players = self.defense_data['nfl_id'].unique()
    
    def calculate_defender_leverage(self, 
                                   off_x: float, off_y: float,
                                   def_x: float, def_y: float,
                                   def_orientation: float) -> Dict:
        """
        Calculate defender's leverage on offensive player
        
        Returns:
            Dict with leverage metrics including:
            - distance: Euclidean distance
            - leverage_angle: Angle advantage/disadvantage
            - cushion: Depth cushion (positive = defender deeper)
            - lateral_leverage: Inside/outside leverage
        """
        distance = np.sqrt((off_x - def_x)**2 + (off_y - def_y)**2)
        
        # Calculate angle from defender to offensive player
        angle_to_receiver = np.degrees(np.arctan2(off_y - def_y, off_x - def_x))
        
        # Leverage angle (how well positioned the defender is)
        leverage_angle = abs(angle_to_receiver - def_orientation)
        leverage_angle = min(leverage_angle, 360 - leverage_angle)  # Take smaller angle
        
        # Cushion (positive = defender has depth)
        cushion = def_x - off_x  
        
        # Lateral leverage (positive = inside, negative = outside)
        field_middle = self.field_width / 2
        lateral_leverage = (field_middle - def_y) - (field_middle - off_y)
        
        return {
            'distance': distance,
            'leverage_angle': leverage_angle,
            'cushion': cushion,
            'lateral_leverage': lateral_leverage,
            'leverage_quality': self._calculate_leverage_quality(distance, leverage_angle, cushion)
        }
    
    def _calculate_leverage_quality(self, distance: float, angle: float, cushion: float) -> float:
        """
        Calculate overall leverage quality (0-1, higher = better defensive leverage)
        """
        # Distance factor (closer is better for defender, normalized)
        dist_factor = max(0, 1 - distance / 15)  # 15 yards = no leverage
        
        # Angle factor (facing receiver is better)
        angle_factor = max(0, 1 - angle / 90)  # 90 degrees = no leverage
        
        # Cushion factor (some cushion is good, too much is bad)
        if cushion < 0:  # Defender beat
            cushion_factor = 0
        elif cushion < 5:  # Ideal cushion
            cushion_factor = 1
        else:  # Too much cushion
            cushion_factor = max(0, 1 - (cushion - 5) / 10)
        
        # Weighted combination
        return 0.4 * dist_factor + 0.3 * angle_factor + 0.3 * cushion_factor
    
    def identify_zone_responsibilities(self, frame_id: int, 
                                      coverage_type: str = 'cover3') -> Dict:
        """
        Identify which defenders are responsible for which zones
        
        Returns:
            Dict mapping zone names to responsible defender IDs
        """
        frame_data = self.defense_data[self.defense_data['frame_id'] == frame_id]
        
        # Get zone definitions
        if coverage_type == 'cover3':
            zones = ZoneCoverage.get_cover3_zones(self.field_width, self.los)
        elif coverage_type == 'cover2':
            zones = ZoneCoverage.get_cover2_zones(self.field_width, self.los)
        else:
            zones = ZoneCoverage.get_cover3_zones(self.field_width, self.los)
        
        zone_assignments = {}
        
        for zone_name, zone_def in zones.items():
            # Find defenders in or near this zone
            defenders_in_zone = []
            
            for _, defender in frame_data.iterrows():
                # Check if defender position matches expected type
                if defender['player_position'] in zone_def['defender_type']:
                    # Check if defender is positioned for this zone
                    x_near = (defender['x'] >= zone_def['x_range'][0] - 3 and 
                             defender['x'] <= zone_def['x_range'][1] + 3)
                    y_near = (defender['y'] >= zone_def['y_range'][0] - 2 and 
                             defender['y'] <= zone_def['y_range'][1] + 2)
                    
                    if x_near and y_near:
                        defenders_in_zone.append({
                            'nfl_id': defender['nfl_id'],
                            'name': defender['player_name'],
                            'position': defender['player_position'],
                            'x': defender['x'],
                            'y': defender['y']
                        })
            
            zone_assignments[zone_name] = defenders_in_zone
        
        return zone_assignments
    
    def calculate_zone_stress(self, frame_id: int, 
                             coverage_type: str = 'cover3') -> Dict:
        """
        Calculate how much stress offensive players put on each zone
        
        Returns:
            Dict with zone stress metrics
        """
        off_frame = self.offense_data[self.offense_data['frame_id'] == frame_id]
        def_frame = self.defense_data[self.defense_data['frame_id'] == frame_id]
        
        # Get zones and assignments
        if coverage_type == 'cover3':
            zones = ZoneCoverage.get_cover3_zones(self.field_width, self.los)
        else:
            zones = ZoneCoverage.get_cover2_zones(self.field_width, self.los)
        
        zone_assignments = self.identify_zone_responsibilities(frame_id, coverage_type)
        
        zone_stress = {}
        
        for zone_name, zone_def in zones.items():
            # Find offensive players in or threatening this zone
            threats = []
            
            for _, receiver in off_frame.iterrows():
                if receiver['player_position'] in ['WR', 'TE', 'RB']:
                    # Check if in zone
                    in_zone = (receiver['x'] >= zone_def['x_range'][0] and 
                              receiver['x'] <= zone_def['x_range'][1] and
                              receiver['y'] >= zone_def['y_range'][0] and 
                              receiver['y'] <= zone_def['y_range'][1])
                    
                    # Check if approaching zone (within 5 yards)
                    near_zone = (receiver['x'] >= zone_def['x_range'][0] - 5 and 
                                receiver['x'] <= zone_def['x_range'][1] + 5 and
                                receiver['y'] >= zone_def['y_range'][0] - 3 and 
                                receiver['y'] <= zone_def['y_range'][1] + 3)
                    
                    if in_zone or near_zone:
                        # Calculate threat level based on speed and direction
                        threat_level = receiver['s'] / 10  # Normalize speed
                        if in_zone:
                            threat_level *= 1.5  # Higher threat if already in zone
                        
                        threats.append({
                            'player': receiver['player_name'],
                            'threat_level': threat_level,
                            'in_zone': in_zone,
                            'speed': receiver['s'],
                            'position': (receiver['x'], receiver['y'])
                        })
            
            # Calculate stress based on threats vs defenders
            defenders = zone_assignments.get(zone_name, [])
            num_defenders = len(defenders)
            num_threats = len(threats)
            total_threat = sum(t['threat_level'] for t in threats)
            
            if num_defenders > 0:
                stress_ratio = total_threat / num_defenders
            else:
                stress_ratio = total_threat * 2 if total_threat > 0 else 0
            
            zone_stress[zone_name] = {
                'defenders': num_defenders,
                'threats': num_threats,
                'total_threat_level': total_threat,
                'stress_ratio': stress_ratio,
                'threat_details': threats,
                'defender_details': defenders
            }
        
        return zone_stress
    
    def calculate_route_synergy(self, frame_id: int, 
                               receiver_ids: List[int]) -> Dict:
        """
        Calculate synergistic effects of multiple receivers
        
        Args:
            frame_id: Frame to analyze
            receiver_ids: List of receiver NFL IDs to analyze together
        
        Returns:
            Dict with synergy metrics
        """
        off_frame = self.offense_data[
            (self.offense_data['frame_id'] == frame_id) & 
            (self.offense_data['nfl_id'].isin(receiver_ids))
        ]
        
        if len(off_frame) < 2:
            return {'synergy_score': 0, 'message': 'Need at least 2 receivers'}
        
        # Get receiver positions
        positions = off_frame[['x', 'y']].values
        
        # Calculate spacing metrics
        distances = cdist(positions, positions)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        
        min_spacing = np.min(distances)
        max_spacing = np.max(distances[distances != np.inf])
        avg_spacing = np.mean(distances[distances != np.inf])
        
        # Calculate coverage area (convex hull)
        if len(positions) >= 3:
            try:
                hull = ConvexHull(positions)
                coverage_area = hull.volume  # In 2D, volume is area
            except:
                coverage_area = 0
        else:
            # For 2 receivers, use distance as "area"
            coverage_area = distances[0, 1] * 5  # Width estimate
        
        # Calculate horizontal and vertical stretch
        x_spread = positions[:, 0].max() - positions[:, 0].min()
        y_spread = positions[:, 1].max() - positions[:, 1].min()
        
        # Check for concept patterns
        concepts = self._identify_route_concepts(off_frame, x_spread, y_spread)
        
        # Calculate synergy score
        synergy_components = {
            'spacing_quality': self._evaluate_spacing(min_spacing, max_spacing, avg_spacing),
            'coverage_area': coverage_area,
            'horizontal_stretch': y_spread,
            'vertical_stretch': x_spread,
            'concept_bonus': concepts['bonus'],
            'concept_identified': concepts['concept']
        }
        
        # Overall synergy score (0-1 scale)
        synergy_score = (
            0.25 * synergy_components['spacing_quality'] +
            0.25 * min(1, coverage_area / 400) +  # Normalize to 400 sq yards
            0.20 * min(1, y_spread / 40) +  # Normalize to 40 yards horizontal
            0.20 * min(1, x_spread / 20) +  # Normalize to 20 yards vertical
            0.10 * concepts['bonus']
        )
        
        return {
            'synergy_score': synergy_score,
            'components': synergy_components,
            'receiver_positions': {
                row['player_name']: (row['x'], row['y']) 
                for _, row in off_frame.iterrows()
            }
        }
    
    def _evaluate_spacing(self, min_spacing: float, max_spacing: float, 
                         avg_spacing: float) -> float:
        """Evaluate spacing quality (0-1 scale)"""
        # Ideal spacing is 8-15 yards apart
        if min_spacing < 5:
            min_factor = min_spacing / 5  # Too close
        elif min_spacing > 20:
            min_factor = max(0, 1 - (min_spacing - 20) / 20)  # Too far
        else:
            min_factor = 1  # Ideal
        
        # Average spacing should be 10-15 yards
        if avg_spacing < 10:
            avg_factor = avg_spacing / 10
        elif avg_spacing > 20:
            avg_factor = max(0, 1 - (avg_spacing - 20) / 15)
        else:
            avg_factor = 1
        
        return 0.6 * min_factor + 0.4 * avg_factor
    
    def _identify_route_concepts(self, receivers_df: pd.DataFrame, 
                                 x_spread: float, y_spread: float) -> Dict:
        """Identify common route concepts that create synergy"""
        concept = "unknown"
        bonus = 0
        
        # Smash concept (high-low)
        if x_spread > 10 and y_spread < 10:
            concept = "smash"
            bonus = 0.3
        
        # Flood concept (3 levels)
        elif len(receivers_df) >= 3 and x_spread > 15:
            concept = "flood"
            bonus = 0.4
        
        # Spacing concept (horizontal stretch)
        elif y_spread > 30 and x_spread < 10:
            concept = "spacing"
            bonus = 0.35
        
        # Four verts (vertical stretch)
        elif x_spread > 20 and y_spread > 35:
            concept = "four_verts"
            bonus = 0.45
        
        # Mesh concept (crossing routes)
        elif 5 < x_spread < 12 and 15 < y_spread < 25:
            concept = "mesh"
            bonus = 0.3
        
        return {'concept': concept, 'bonus': bonus}
    
    def calculate_coverage_burden(self, frame_id: int, defender_id: int) -> Dict:
        """
        Calculate the coverage burden on a specific defender
        
        Returns:
            Dict with burden metrics including area to cover and threats
        """
        def_row = self.defense_data[
            (self.defense_data['frame_id'] == frame_id) & 
            (self.defense_data['nfl_id'] == defender_id)
        ].iloc[0]
        
        off_frame = self.offense_data[
            (self.offense_data['frame_id'] == frame_id) & 
            (self.offense_data['player_position'].isin(['WR', 'TE', 'RB']))
        ]
        
        # Find receivers in defender's area (within 15 yards)
        threats = []
        for _, receiver in off_frame.iterrows():
            distance = np.sqrt((def_row['x'] - receiver['x'])**2 + 
                             (def_row['y'] - receiver['y'])**2)
            
            if distance <= 15:
                leverage = self.calculate_defender_leverage(
                    receiver['x'], receiver['y'],
                    def_row['x'], def_row['y'],
                    def_row['o']
                )
                
                threats.append({
                    'receiver': receiver['player_name'],
                    'distance': distance,
                    'leverage_quality': leverage['leverage_quality'],
                    'speed': receiver['s']
                })
        
        # Sort threats by proximity and speed
        threats = sorted(threats, key=lambda x: x['distance'] - x['speed']*0.5)
        
        # Calculate burden score
        if not threats:
            burden_score = 0
        else:
            # Primary threat contributes most to burden
            primary_burden = (1 - threats[0]['leverage_quality']) * 0.6
            
            # Additional threats add burden
            secondary_burden = sum(
                (1 - t['leverage_quality']) * 0.2 
                for t in threats[1:3]  # Consider up to 2 more threats
            )
            
            burden_score = min(1, primary_burden + secondary_burden)
        
        return {
            'defender': def_row['player_name'],
            'position': def_row['player_position'],
            'num_threats': len(threats),
            'burden_score': burden_score,
            'threats': threats,
            'location': (def_row['x'], def_row['y'])
        }
    
    def analyze_full_play(self, frame_id: int, coverage_type: str = 'cover3') -> Dict:
        """
        Complete analysis of offensive positioning vs defensive coverage
        
        Returns:
            Comprehensive dict with all metrics
        """
        # Get all eligible receivers
        receivers = self.offense_data[
            (self.offense_data['frame_id'] == frame_id) & 
            (self.offense_data['player_position'].isin(['WR', 'TE', 'RB']))
        ]['nfl_id'].unique()
        
        # Zone stress analysis
        zone_stress = self.calculate_zone_stress(frame_id, coverage_type)
        
        # Route synergy for all receivers
        synergy = self.calculate_route_synergy(frame_id, receivers.tolist())
        
        # Individual defender burdens
        defender_burdens = {}
        for def_id in self.defensive_players:
            defender_burdens[def_id] = self.calculate_coverage_burden(frame_id, def_id)
        
        # Calculate overall offensive advantage
        total_zone_stress = sum(z['stress_ratio'] for z in zone_stress.values())
        avg_defender_burden = np.mean([d['burden_score'] for d in defender_burdens.values()])
        
        offensive_advantage = (
            0.4 * synergy['synergy_score'] +
            0.3 * min(1, total_zone_stress / 10) +  # Normalize stress
            0.3 * avg_defender_burden
        )
        
        return {
            'frame_id': frame_id,
            'coverage_type': coverage_type,
            'offensive_advantage': offensive_advantage,
            'zone_stress': zone_stress,
            'route_synergy': synergy,
            'defender_burdens': defender_burdens,
            'summary': {
                'total_zone_stress': total_zone_stress,
                'average_defender_burden': avg_defender_burden,
                'synergy_score': synergy['synergy_score'],
                'stressed_zones': sum(1 for z in zone_stress.values() if z['stress_ratio'] > 1),
                'overloaded_defenders': sum(1 for d in defender_burdens.values() if d['burden_score'] > 0.7)
            }
        }