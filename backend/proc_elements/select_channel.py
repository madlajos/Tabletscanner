import cv2
import numpy as np


def select_channel(data, space="BGR", channel="R", debug=False):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E2100"
        return data

    channel_map = {
        "BGR": {"B": 0, "G": 1, "R": 2},
        "HSV": {"H": 0, "S": 1, "V": 2},
        "LAB": {"L": 0, "A": 1, "B": 2},
        "GRAY": {"GRAY": 0}
    }

    output_images = []

    for img in data["images"]:

        if img is None:
            data["error"] = "E2103"
            return data

        if space == "GRAY":
            if len(img.shape) == 2:
                converted = img
            elif len(img.shape) == 3 and img.shape[2] == 3:
                converted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                data["error"] = "E2104"
                return data

            channel_img = converted

        else:
            if len(img.shape) != 3 or img.shape[2] != 3:
                data["error"] = "E2104"
                return data

            if space == "BGR":
                converted = img
            elif space == "HSV":
                converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif space == "LAB":
                converted = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            else:
                data["error"] = "E2101"
                return data

            # Handle "ALL" option: stack all three channels back into a 3-channel image
            if channel == "ALL":
                # Convert back to BGR for preview/display purposes
                if space == "HSV":
                    channel_img = cv2.cvtColor(converted, cv2.COLOR_HSV2BGR)
                elif space == "LAB":
                    channel_img = cv2.cvtColor(converted, cv2.COLOR_LAB2BGR)
                else:
                    channel_img = converted
            else:
                idx = channel_map[space].get(channel)
                if idx is None:
                    data["error"] = "E2102"
                    return data

                channel_img = converted[:, :, idx]

        output_images.append(channel_img)

    data["images"] = output_images
    data["count"] = len(output_images)
    data["meta"]["channel"] = {
        "space": space,
        "channel": channel
    }
    # Also store with select_channel key for pipeline consumption
    data["meta"]["select_channel"] = {
        "space": space,
        "channel": channel
    }
    data["history"].append("select_channel")

    if debug:
        print(data["meta"]["channel"])
        print(f"Output shape: {output_images[0].shape}")

    return data
