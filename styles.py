
"""
Reward definitions for different learning styles.
"""

STYLES = {
    "Standard (Curriculum)": {
        "type": "curriculum",
        "description": "Progressive learning: Food -> Walls -> Enemies -> Strategy",
        "stages": {
            1: {
                "name": "FOOD_VECTOR",
                "gamma": 0.9593237458769065,
                "food_reward": 1.0872386222509225,
                "food_shaping": 0.0010958841672559167,
                "survival": 0.0030155707383473316,
                "death_wall": -15.552218893461756,
                "death_snake": -5,
                "wall_proximity_penalty": 0.005556018479145987,
                "max_steps": 823,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_food": 12,
                    "avg_steps": 80
                },
                "promote_window": 400
            },
            2: {
                "name": "WALL_AVOID",
                "gamma": 0.9832002446278482,
                "food_reward": 3.1566677075801217,
                "food_shaping": 0.026178633283812147,
                "survival": 0.02316152374218733,
                "survival_escalation": 0.00011112905308528195,
                "death_wall": -6.911567212754948,
                "death_snake": -7.666960876176105,
                "wall_alert_dist": 2500,
                "wall_proximity_penalty": 0.0002963534422125578,
                "starvation_penalty": 7.62728594431461e-05,
                "starvation_grace_steps": 5,
                "max_steps": 180,
                "promote_metric": "avg_steps",
                "promote_threshold": 120,
                "promote_wall_death_max": 0.1,
                "promote_window": 400
            },
            3: {
                "name": "ENEMY_AVOID",
                "gamma": 0.9290821047411956,
                "food_reward": 3.7534694785699796,
                "food_shaping": 0.0899568440140703,
                "survival": 0.30545305396530975,
                "death_wall": -5.29372182340461,
                "death_snake": -42.18616130295092,
                "enemy_alert_dist": 2395,
                "enemy_proximity_penalty": 1.1235791263555546,
                "enemy_approach_penalty": 0.8865735074526992,
                "boost_penalty": 0.07557440472126242,
                "starvation_penalty": 0.0013842901197462423,
                "starvation_grace_steps": 48,
                "max_steps": 842,
                "promote_metric": "avg_steps",
                "promote_threshold": 350,
                "promote_window": 500
            },
            4: {
                "name": "MASS_MANAGEMENT",
                "gamma": 0.9345310066275695,
                "food_reward": 11.622073679357989,
                "food_shaping": 0.0810601559799751,
                "survival": 0.00935123892817459,
                "survival_escalation": 0.00047205121862332064,
                "death_wall": -33.55856342463639,
                "death_snake": -22.732783640375693,
                "length_bonus": 0.002180725523050778,
                "wall_proximity_penalty": 0.01724094920698024,
                "enemy_alert_dist": 1388,
                "enemy_proximity_penalty": 0.04942497726690352,
                "enemy_approach_penalty": 0.0010861328654893014,
                "boost_penalty": 0.0004698346836458764,
                "mass_loss_penalty": 0.06500233157590378,
                "starvation_penalty": 0.00010774927977767561,
                "starvation_grace_steps": 8,
                "max_steps": 1749,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_steps": 600,
                    "avg_peak_length": 40
                },
                "promote_window": 200
            },
            5: {
                "name": "MASTERY_SURVIVAL",
                "gamma": 0.995,
                "food_reward": 3.878435206593664,
                "food_shaping": 0.09269246289894084,
                "survival": 0.07936050326806246,
                "survival_escalation": 0.0019461765396040036,
                "death_wall": -23.664030727188006,
                "death_snake": -31.687700422432613,
                "length_bonus": 0.0060892529642129,
                "wall_proximity_penalty": 0.38132229474966967,
                "enemy_alert_dist": 1329,
                "enemy_proximity_penalty": 1.7712159327969892,
                "enemy_approach_penalty": 0.10344173758026577,
                "boost_penalty": 6.047500084734265,
                "mass_loss_penalty": 1.8904595251282406,
                "starvation_penalty": 0.011613285062345293,
                "starvation_grace_steps": 138,
                "contest_food_reward": 0.9186165032542355,
                "enemy_zone_control_reward": 0.06585261178978555,
                "kill_opportunity_reward": 10.526338927984318,
                "max_steps": 4035,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_steps": 1200,
                    "avg_peak_length": 100
                },
                "promote_window": 500
            },
            6: {
                "name": "APEX_PREDATOR",
                "gamma": 0.99,
                "food_reward": 8.0,
                "survival": 0.1,
                "death_wall": -40,
                "death_snake": -30,
                "enemy_alert_dist": 2000,
                "enemy_proximity_penalty": 0.15,
                "boost_penalty": 0.0,
                "contest_food_reward": 1.0,
                "enemy_zone_control_reward": 0.06,
                "kill_opportunity_reward": 18.0,
                "max_steps": 99999,
                "promote_metric": None
            }
        }
    },
    "Aggressive (Hunter)": {
        "type": "static",
        "description": "High reward for eating. Low survival bonus.",
        "config": {
            "name": "HUNTER",
            "food_reward": 20.0,
            "food_shaping": 0.05,
            "survival": 0.0,
            "death_wall": -50,
            "death_snake": -10,
            "wall_proximity_penalty": 0.05,
            "enemy_proximity_penalty": 0.05,
            "max_steps": 99999
        }
    },
    "Defensive (Safe)": {
        "type": "static",
        "description": "High survival bonus and heavy death penalties.",
        "config": {
            "name": "SAFE",
            "food_reward": 5.0,
            "food_shaping": 0.005,
            "survival": 0.5,
            "death_wall": -50,
            "death_snake": -40,
            "straight_penalty": 0.05,
            "wall_proximity_penalty": 0.2,
            "enemy_proximity_penalty": 0.15,
            "max_steps": 99999
        }
    },
    "Explorer (Anti-Float)": {
        "type": "static",
        "description": "Penalizes staying still. Forces movement.",
        "config": {
            "name": "EXPLORER",
            "food_reward": 5.0,
            "food_shaping": 0.05,
            "survival": 0.05,
            "death_wall": -50,
            "death_snake": -40,
            "straight_penalty": 0.1,
            "wall_proximity_penalty": 0.5,
            "enemy_proximity_penalty": 0.1,
            "max_steps": 99999
        }
    }
}
