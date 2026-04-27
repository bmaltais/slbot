
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
                "gamma": 0.9086652951894412,
                "food_reward": 4.459557788388153,
                "food_shaping": 4.006857745302811e-05,
                "survival": 0.004790632200456803,
                "death_wall": -13.805999963368029,
                "death_snake": -5,
                "wall_proximity_penalty": 0.00979067318143637,
                "boost_penalty": 0.21442822533506145,
                "mass_loss_penalty": 0.18400489066117912,
                "max_steps": 612,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_food": 12,
                    "avg_steps": 80
                },
                "promote_window": 400
            },
            2: {
                "name": "WALL_AVOID",
                "gamma": 0.9183433591238422,
                "food_reward": 1.0,
                "food_shaping": 0.002092677359157219,
                "survival": 0.025525260163423567,
                "survival_escalation": 0.00024640839284192955,
                "death_wall": -9.764715039903807,
                "death_snake": -5,
                "wall_alert_dist": 2500,
                "wall_proximity_penalty": 0.00024089975759018404,
                "boost_penalty": 0.929601590472811,
                "mass_loss_penalty": 0.0944500566960564,
                "starvation_penalty": 0.00014237769270900613,
                "starvation_grace_steps": 5,
                "max_steps": 331,
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
                "boost_penalty": 1.0,
                "mass_loss_penalty": 0.5,
                "starvation_penalty": 0.0013842901197462423,
                "starvation_grace_steps": 48,
                "max_steps": 842,
                "promote_metric": "avg_steps",
                "promote_threshold": 350,
                "promote_window": 500
            },
            4: {
                "name": "MASS_MANAGEMENT",
                "gamma": 0.995,
                "food_reward": 8.159769269305478,
                "food_shaping": 0.007306359235764289,
                "survival": 0.008960139865825922,
                "survival_escalation": 5.562378300603925e-05,
                "death_wall": -37.345252462476765,
                "death_snake": -15.958259317025277,
                "length_bonus": 0.054085741339848906,
                "wall_proximity_penalty": 0.05177598004353546,
                "enemy_alert_dist": 894,
                "enemy_proximity_penalty": 0.032381343570352925,
                "enemy_approach_penalty": 0.0014217337981085453,
                "boost_penalty": 1.5786085762426325,
                "mass_loss_penalty": 4.050152594603316,
                "starvation_penalty": 4.759217472556548e-05,
                "starvation_grace_steps": 6,
                "max_steps": 2677,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_steps": 600,
                    "avg_peak_length": 60
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
                "length_bonus": 0.1,
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
                "kill_opportunity_reward": 3.0,
                "max_steps": 4035,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_steps": 1200,
                    "avg_peak_length": 80
                },
                "promote_window": 500
            },
            6: {
                "name": "APEX_PREDATOR",
                "gamma": 0.995,
                "food_reward": 5.615299111119233,
                "survival": 0.1789453656119041,
                "death_wall": -52.80746300589298,
                "death_snake": -58.01156974535044,
                "enemy_alert_dist": 1214,
                "enemy_proximity_penalty": 0.01254203936732087,
                "boost_penalty": 2.0,
                "mass_loss_penalty": 1.0,
                "length_bonus": 0.1,
                "contest_food_reward": 0.09320524639420258,
                "enemy_zone_control_reward": 0.0007241584567977896,
                "kill_opportunity_reward": 2.5,
                "max_steps": 9302,
                "promote_metric": None,
                "enemy_approach_penalty": 0.454704195060626,
                "promote_window": 500,
                "promote_conditions": {},
                "promote_wall_death_max": 1.0
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
