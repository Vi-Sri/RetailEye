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
    

    def detect_barcode_switch(self, detected_objects,objects_confidences, barcode_number, take_top_n=1) -> bool:
        """
        Detects if barcode is switched based on barcode numebr and detected object.
        """
        # print("In barcode switch")

        IS_TICKET_SWTICH = False
        TICKET_SWTICH_COUNTER = 0
        product_data = None

        try:
            product_data = self.product_database.get(barcode_number, None)
        except KeyError:
            print(f"No product data for: {barcode_number}")
            return None
        
        # print("before product data is not none")

        if product_data is not None:

            # print("product_data", product_data)

            product_category = product_data['product_category']
            product_price = product_data['price']
            product_name = product_data['name']

            product_name = "FiberGummies"  #uncomment to check ticket switch manually

            print("Barcode product: ", product_name, "Object classification", detected_objects, objects_confidences)


            if len(detected_objects)>=take_top_n:
                # print(detected_objects, objects_confidences)
                take_top_n +=1
                filtered_detected_objects, filtered_objects_confidences = detected_objects[:take_top_n], objects_confidences[:take_top_n]

                # print(filtered_detected_objects, filtered_objects_confidences)
                # print("Barcode product: ", product_name, "Object classification", filtered_detected_objects, filtered_objects_confidences)
                
                for idx, det_obj in enumerate(filtered_detected_objects):
                    # print(idx, det_obj)
                    if idx==0:
                        if det_obj=="UnknownObjects": # No need to check ticketswitch
                            break
                    
                    if det_obj!=product_name:
                        TICKET_SWTICH_COUNTER +=1
                    else:
                        break

                if TICKET_SWTICH_COUNTER>=take_top_n-1:
                    IS_TICKET_SWTICH = True
            else:
                print(f"no object detected len: {len(detected_objects)}")


            

            # print("TICKET_SWTICH_COUNTER:", TICKET_SWTICH_COUNTER)

        else:
            print(f"product_data is none for : {barcode_number}")


        return IS_TICKET_SWTICH