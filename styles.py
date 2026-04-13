
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
                "gamma": 0.9153498292370831,
                "food_reward": 3.150860819028675,
                "food_shaping": 6.0317540610334526e-05,
                "survival": 0.00265578247148251,
                "death_wall": -13.893671302356353,
                "death_snake": -5,
                "wall_proximity_penalty": 0.0036957856964702014,
                "max_steps": 810,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_food": 12,
                    "avg_steps": 80
                },
                "promote_window": 400
            },
            2: {
                "name": "WALL_AVOID",
                "gamma": 0.9750325549707558,
                "food_reward": 1.2147597890020467,
                "food_shaping": 0.004095592308209608,
                "survival": 0.024398253497298957,
                "survival_escalation": 0.00024640839284192955,
                "death_wall": -9.764715039903807,
                "death_snake": -6.809996481564172,
                "wall_alert_dist": 2500,
                "wall_proximity_penalty": 0.00013233021656861398,
                "starvation_penalty": 0.00011644347817704535,
                "starvation_grace_steps": 6,
                "max_steps": 195,
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
                "gamma": 0.9330286796433835,
                "food_reward": 7.100206802039844,
                "food_shaping": 0.019277595545402683,
                "survival": 0.013078108196244158,
                "survival_escalation": 0.000583317488698864,
                "death_wall": -37.269217559969356,
                "death_snake": -19.125811540821704,
                "length_bonus": 0.0006184221853482939,
                "wall_proximity_penalty": 0.030533893832520285,
                "enemy_alert_dist": 1492,
                "enemy_proximity_penalty": 0.032381343570352925,
                "enemy_approach_penalty": 0.0014217337981085453,
                "boost_penalty": 0.0007995080726862225,
                "mass_loss_penalty": 0.028044249928529795,
                "starvation_penalty": 4.070573811273068e-05,
                "starvation_grace_steps": 11,
                "max_steps": 2542,
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
                "gamma": 0.954365166852315,
                "food_reward": 5.124071519018861,
                "survival": 0.14660911797419837,
                "death_wall": -46.207292078772696,
                "death_snake": -100,
                "enemy_alert_dist": 1504,
                "enemy_proximity_penalty": 0.04025590489973749,
                "boost_penalty": 0.009267192406595667,
                "contest_food_reward": 0.2701115523846602,
                "enemy_zone_control_reward": 0.0009168878092246559,
                "kill_opportunity_reward": 8.550420450984676,
                "max_steps": 7260,
                "promote_metric": None,
                "enemy_approach_penalty": 1.259783648839948,
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
