from dataclasses import dataclass
from typing import Dict

@dataclass
class ZoneCoverage:
    """Defines standard zone coverage areas on the field"""
    
    @staticmethod
    def get_cover3_zones(field_width: float, los: float) -> Dict:
        """
        Define Cover 3 zone responsibilities
        Field is 53.3 yards wide, zones defined from line of scrimmage

        Args:
            field_width: Width of the field in yards (typically 53.3)
            los: Line of scrimmage position
        """
        zones = {
            # Deep thirds (15+ yards from LOS)
            'deep_left': {
                'x_range': (los, los + 30),
                'y_range': (0, field_width/3),
                'defender_type': ['CB', 'FS'],
                'depth': 'deep'
            },
            'deep_middle': {
                'x_range': (los, los + 30),
                'y_range': (field_width/3, 2*field_width/3),
                'defender_type': ['FS'],
                'depth': 'deep'
            },
            'deep_right': {
                'x_range': (los, los + 30),
                'y_range': (2*field_width/3, field_width),
                'defender_type': ['CB', 'SS'],
                'depth': 'deep'
            },
            # Intermediate zones (5-15 yards)
            'hook_curl_left': {
                'x_range': (los + 5, los + 15),
                'y_range': (5, field_width/3),
                'defender_type': ['OLB', 'CB'],
                'depth': 'intermediate'
            },
            'middle_hole': {
                'x_range': (los + 5, los + 15),
                'y_range': (field_width/3, 2*field_width/3),
                'defender_type': ['ILB'],
                'depth': 'intermediate'
            },
            'hook_curl_right': {
                'x_range': (los + 5, los + 15),
                'y_range': (2*field_width/3, field_width - 5),
                'defender_type': ['OLB', 'CB'],
                'depth': 'intermediate'
            },
            # Underneath zones (0-5 yards)
            'flat_left': {
                'x_range': (los, los + 5),
                'y_range': (0, field_width/3),
                'defender_type': ['OLB', 'CB'],
                'depth': 'shallow'
            },
            'flat_right': {
                'x_range': (los, los + 5),
                'y_range': (2*field_width/3, field_width),
                'defender_type': ['OLB', 'CB'],
                'depth': 'shallow'
            }
        }
        return zones
    
    @staticmethod
    def get_cover2_zones(field_width: float = 53.3, los: float = 46) -> Dict:
        """Define Cover 2 zone responsibilities"""
        zones = {
            'deep_left': {
                'x_range': (los + 12, los + 30),
                'y_range': (0, field_width/2),
                'defender_type': ['FS', 'SS'],
                'depth': 'deep'
            },
            'deep_right': {
                'x_range': (los + 12, los + 30),
                'y_range': (field_width/2, field_width),
                'defender_type': ['FS', 'SS'],
                'depth': 'deep'
            },
            # 5 underneath zones
            'flat_left': {
                'x_range': (los, los + 7),
                'y_range': (0, field_width * 0.2),
                'defender_type': ['CB'],
                'depth': 'shallow'
            },
            'hook_left': {
                'x_range': (los + 5, los + 12),
                'y_range': (field_width * 0.2, field_width * 0.4),
                'defender_type': ['OLB'],
                'depth': 'intermediate'
            },
            'middle': {
                'x_range': (los + 3, los + 12),
                'y_range': (field_width * 0.4, field_width * 0.6),
                'defender_type': ['ILB'],
                'depth': 'intermediate'
            },
            'hook_right': {
                'x_range': (los + 5, los + 12),
                'y_range': (field_width * 0.6, field_width * 0.8),
                'defender_type': ['OLB'],
                'depth': 'intermediate'
            },
            'flat_right': {
                'x_range': (los, los + 7),
                'y_range': (field_width * 0.8, field_width),
                'defender_type': ['CB'],
                'depth': 'shallow'
            }
        }
        return zones