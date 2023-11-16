class BarcodeSwitchDetector:

    def __init__(self, csv_db_file_path) -> None:
        self.barcode_mapping = {}
        self.load_barcode_dictionary(csv_path=csv_db_file_path)
        pass


    def load_barcode_dictionary(self, csv_path):
        lines = []
        with open(csv_path) as f:
            lines = f.readlines()

        for line in enumerate(lines):
            line = line.strip()
            barcode, productname = line.split(",")
            if not barcode in self.barcode_mapping:
                self.barcode_mapping[barcode] = productname
    

    def detect_barcode_switch(self, detected_objects,objects_confidences, barcode_number, take_top_n=3) -> bool:
        """
        Detects if barcode is switched based on barcode numebr and detected object.
        """

        IS_TICKET_SWTICH = False
        TICKET_SWTICH_COUNTER = 0

        product_name = self.barcode_mapping[barcode_number]

        filtered_detected_objects, filtered_objects_confidences = self.sort_and_filter_objects(detected_objects,objects_confidences, take_top_n=take_top_n)

        for idx, det_obj in enumerate(filtered_detected_objects):
            if det_obj!=product_name:
                TICKET_SWTICH_COUNTER +=1

        if TICKET_SWTICH_COUNTER==take_top_n:
            IS_TICKET_SWTICH = True

        return IS_TICKET_SWTICH


    def sort_and_filter_objects(self, detected_objects,objects_confidences, take_top_n=3):
        """
        TODO sort objects based on bounding box and confidence. take top n.
        """
        filtered_detected_objects, filtered_objects_confidences = None, None

        return filtered_detected_objects, filtered_objects_confidences