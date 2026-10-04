class WorldModel:
    """Lưu giữ ngữ cảnh môi trường (Player, Game Context giống thiết kế Astra)"""
    def __init__(self):
        self.state = {"status": "idle", "current_task": "none", "location": "Render Web Service"}
        
    def update(self, key, value):
        self.state[key] = value

world_model = WorldModel:()
