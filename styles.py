
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
                "gamma": 0.8839791669561878,
                "food_reward": 3.4435852643587577,
                "food_shaping": 4.006857745302811e-05,
                "survival": 0.006256031207020906,
                "death_wall": -11.156581759253891,
                "death_snake": -5.90540994990689,
                "wall_proximity_penalty": 0.004981493540786513,
                "boost_penalty": 0.7675109141524663,
                "mass_loss_penalty": 0.18400489066117912,
                "max_steps": 660,
                "promote_metric": "compound",
                "promote_conditions": {
                    "avg_food": 12,
                    "avg_steps": 80
                },
                "promote_window": 400
            },
            2: {
                "name": "WALL_AVOID",
                "gamma": 0.9449688730903726,
                "food_reward": 1.0,
                "food_shaping": 0.0029400314872170584,
                "survival": 0.022304735397887288,
                "survival_escalation": 0.00024640839284192955,
                "death_wall": -9.764715039903807,
                "death_snake": -5.014264598235712,
                "wall_alert_dist": 2500,
                "wall_proximity_penalty": 0.000143122772215361,
                "boost_penalty": 0.7697389582849905,
                "mass_loss_penalty": 0.247716407020985,
                "starvation_penalty": 0.00011644347817704535,
                "starvation_grace_steps": 5,
                "max_steps": 261,
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
                "gamma": 0.9233214337642229,
                "food_reward": 8.159769269305478,
                "food_shaping": 0.010796593393108556,
                "survival": 0.007153736903799961,
                "survival_escalation": 0.0001853603688545922,
                "death_wall": -37.269217559969356,
                "death_snake": -19.125811540821704,
                "length_bonus": 0.05,
                "wall_proximity_penalty": 0.03633055975932782,
                "enemy_alert_dist": 1447,
                "enemy_proximity_penalty": 0.032381343570352925,
                "enemy_approach_penalty": 0.0014217337981085453,
                "boost_penalty": 2.946305049045784,
                "mass_loss_penalty": 2.3266652446331273,
                "starvation_penalty": 2.3772468680138982e-05,
                "starvation_grace_steps": 7,
                "max_steps": 2600,
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
