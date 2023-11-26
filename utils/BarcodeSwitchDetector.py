import json

class BarcodeSwitchDetector:

    def __init__(self, db_file_path, isJson=True) -> None:
        self.product_database = {}
        # self.load_barcode_dictionary(csv_path=csv_db_file_path)
        if isJson:
            self.load_barcode_db(json_file_path=db_file_path)
        else:
            self.load_barcode_dictionary(db_file_path)

    def load_barcode_db(self, json_file_path):
        # Loading product database
        with open(json_file_path, 'r') as f:
            self.product_database = json.load(f)

    def load_barcode_dictionary(self, csv_path):
        lines = []
        with open(csv_path) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            # print(line)
            line = line.strip()
            barcode, productname = line.split(",")
            if not barcode in self.product_database:
                self.product_database[barcode] = productname
    

    def detect_barcode_switch(self, detected_objects,objects_confidences, barcode_number, take_top_n=3) -> bool:
        """
        Detects if barcode is switched based on barcode numebr and detected object.
        """

        IS_TICKET_SWTICH = False
        TICKET_SWTICH_COUNTER = 0

        product_name = self.product_database[barcode_number]

        filtered_detected_objects, filtered_objects_confidences = detected_objects[:take_top_n], objects_confidences[:take_top_n]

        for idx, det_obj in enumerate(filtered_detected_objects):
            if det_obj!=product_name:
                TICKET_SWTICH_COUNTER +=1

        if TICKET_SWTICH_COUNTER>=take_top_n:
            IS_TICKET_SWTICH = True

        return IS_TICKET_SWTICH