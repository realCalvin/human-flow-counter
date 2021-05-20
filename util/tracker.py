import math

class EuclideanDistTracker:
    def __init__(self):
        # center positions of the objects
        self.center_points = {}
        
        # current unique id count
        self.id_count = 0


    def update(self, objects):
        # stored objects/ids
        objects_id = []

        # centroid of object
        for box in objects:
            x, y, w, h, confidence = box
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # determine if object exists
            detected_object = False
            for id, point in self.center_points.items():

                # euclidean distance
                dist = math.hypot(cx - point[0], cy - point[1])

                # euclidean distance threshold
                if dist < 100:
                    self.center_points[id] = (cx, cy)
                    objects_id.append([x, y, w, h, id, confidence])
                    detected_object = True
                    break

            # new object detected
            if detected_object is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_id.append([x, y, w, h, self.id_count, confidence])
                self.id_count += 1

        # remove ids that are no longer used
        temp_center_points = {}
        for obj in objects_id:
            _, _, _, _, object_id, _ = obj
            center = self.center_points[object_id]
            temp_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = temp_center_points.copy()
        return objects_id