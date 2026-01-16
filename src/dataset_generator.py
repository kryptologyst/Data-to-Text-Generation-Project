"""
Synthetic dataset generation for data-to-text training and evaluation.
"""

import json
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from faker import Faker

from .data_to_text_generator import StructuredData


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    num_samples: int = 1000
    categories: List[str] = None
    output_dir: str = "data"
    random_seed: int = 42
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = ["weather", "product", "company", "person", "event"]


class SyntheticDatasetGenerator:
    """Generate synthetic datasets for data-to-text generation."""
    
    def __init__(self, config: DatasetConfig):
        """Initialize the dataset generator."""
        self.config = config
        self.fake = Faker()
        random.seed(config.random_seed)
        self.fake.seed(config.random_seed)
    
    def generate_weather_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic weather data."""
        data = []
        
        cities = [
            "New York", "London", "Paris", "Tokyo", "Sydney", "Berlin", 
            "Madrid", "Rome", "Amsterdam", "Barcelona", "Vienna", "Prague"
        ]
        
        conditions = ["sunny", "cloudy", "rainy", "snowy", "foggy", "stormy"]
        
        for _ in range(num_samples):
            city = random.choice(cities)
            temp = random.randint(-10, 40)
            humidity = random.randint(30, 90)
            condition = random.choice(conditions)
            wind_speed = random.randint(5, 30)
            
            data.append({
                "city": city,
                "temperature": f"{temp}°C",
                "humidity": f"{humidity}%",
                "forecast": condition,
                "wind_speed": f"{wind_speed} km/h",
                "category": "weather"
            })
        
        return data
    
    def generate_product_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic product data."""
        data = []
        
        products = [
            "iPhone 15", "Samsung Galaxy S24", "MacBook Pro", "Dell XPS 13",
            "iPad Air", "Surface Pro", "AirPods Pro", "Sony WH-1000XM5",
            "Nintendo Switch", "PlayStation 5", "Tesla Model 3", "BMW i4"
        ]
        
        colors = ["Black", "White", "Silver", "Gold", "Blue", "Red", "Green"]
        storage_options = ["64GB", "128GB", "256GB", "512GB", "1TB"]
        
        for _ in range(num_samples):
            product = random.choice(products)
            price = random.randint(100, 2000)
            storage = random.choice(storage_options)
            color = random.choice(colors)
            availability = random.choice(["In Stock", "Out of Stock", "Limited Stock"])
            
            data.append({
                "product": product,
                "price": f"${price}",
                "storage": storage,
                "color": color,
                "availability": availability,
                "category": "product"
            })
        
        return data
    
    def generate_company_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic company data."""
        data = []
        
        industries = ["Technology", "Finance", "Healthcare", "Manufacturing", "Retail", "Education"]
        locations = ["San Francisco", "New York", "London", "Berlin", "Tokyo", "Sydney"]
        
        for _ in range(num_samples):
            company = self.fake.company()
            revenue = random.randint(1, 100)  # Million
            employees = random.randint(10, 10000)
            industry = random.choice(industries)
            location = random.choice(locations)
            
            data.append({
                "company": company,
                "revenue": f"${revenue}M",
                "employees": str(employees),
                "industry": industry,
                "location": location,
                "category": "company"
            })
        
        return data
    
    def generate_person_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic person data."""
        data = []
        
        professions = ["Engineer", "Doctor", "Teacher", "Lawyer", "Designer", "Manager"]
        
        for _ in range(num_samples):
            name = self.fake.name()
            age = random.randint(22, 65)
            profession = random.choice(professions)
            city = self.fake.city()
            salary = random.randint(30000, 200000)
            
            data.append({
                "name": name,
                "age": str(age),
                "profession": profession,
                "city": city,
                "salary": f"${salary:,}",
                "category": "person"
            })
        
        return data
    
    def generate_event_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic event data."""
        data = []
        
        event_types = ["Conference", "Workshop", "Concert", "Exhibition", "Seminar", "Festival"]
        venues = ["Convention Center", "Hotel", "Park", "Theater", "Stadium", "Museum"]
        
        for _ in range(num_samples):
            event_name = f"{random.choice(event_types)} {self.fake.word().title()}"
            date = self.fake.date_between(start_date='-1y', end_date='+1y')
            venue = random.choice(venues)
            city = self.fake.city()
            attendees = random.randint(50, 5000)
            
            data.append({
                "event": event_name,
                "date": str(date),
                "venue": venue,
                "city": city,
                "attendees": str(attendees),
                "category": "event"
            })
        
        return data
    
    def generate_dataset(self) -> List[Dict[str, Any]]:
        """Generate the complete synthetic dataset."""
        all_data = []
        samples_per_category = self.config.num_samples // len(self.config.categories)
        
        for category in self.config.categories:
            if category == "weather":
                data = self.generate_weather_data(samples_per_category)
            elif category == "product":
                data = self.generate_product_data(samples_per_category)
            elif category == "company":
                data = self.generate_company_data(samples_per_category)
            elif category == "person":
                data = self.generate_person_data(samples_per_category)
            elif category == "event":
                data = self.generate_event_data(samples_per_category)
            else:
                continue
            
            all_data.extend(data)
        
        # Shuffle the data
        random.shuffle(all_data)
        
        return all_data
    
    def create_reference_texts(self, data: List[Dict[str, Any]]) -> List[str]:
        """Create reference texts for evaluation."""
        reference_texts = []
        
        for item in data:
            category = item.get("category", "")
            
            if category == "weather":
                ref_text = (
                    f"The weather in {item['city']} is {item['forecast']} "
                    f"with a temperature of {item['temperature']}, "
                    f"humidity at {item['humidity']}, and wind speed of {item['wind_speed']}."
                )
            elif category == "product":
                ref_text = (
                    f"The {item['product']} is available in {item['color']} "
                    f"with {item['storage']} storage for {item['price']}. "
                    f"Current availability: {item['availability']}."
                )
            elif category == "company":
                ref_text = (
                    f"{item['company']} is a {item['industry']} company "
                    f"located in {item['location']} with {item['employees']} employees "
                    f"and annual revenue of {item['revenue']}."
                )
            elif category == "person":
                ref_text = (
                    f"{item['name']} is a {item['age']}-year-old {item['profession']} "
                    f"living in {item['city']} with an annual salary of {item['salary']}."
                )
            elif category == "event":
                ref_text = (
                    f"{item['event']} will take place on {item['date']} "
                    f"at {item['venue']} in {item['city']} "
                    f"with an expected attendance of {item['attendees']} people."
                )
            else:
                ref_text = "Generated text for structured data."
            
            reference_texts.append(ref_text)
        
        return reference_texts
    
    def save_dataset(self, data: List[Dict[str, Any]], reference_texts: List[str]) -> None:
        """Save the generated dataset to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(output_dir / "synthetic_dataset.json", 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(data)
        df.to_csv(output_dir / "synthetic_dataset.csv", index=False)
        
        # Save reference texts
        with open(output_dir / "reference_texts.json", 'w') as f:
            json.dump(reference_texts, f, indent=2)
        
        # Save dataset info
        dataset_info = {
            "num_samples": len(data),
            "categories": list(set(item["category"] for item in data)),
            "generation_config": {
                "random_seed": self.config.random_seed,
                "num_samples": self.config.num_samples
            }
        }
        
        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset saved to {output_dir}")
        print(f"Generated {len(data)} samples across {len(dataset_info['categories'])} categories")


def main():
    """Generate synthetic dataset for demonstration."""
    config = DatasetConfig(
        num_samples=500,
        categories=["weather", "product", "company", "person", "event"]
    )
    
    generator = SyntheticDatasetGenerator(config)
    
    print("Generating synthetic dataset...")
    data = generator.generate_dataset()
    reference_texts = generator.create_reference_texts(data)
    
    generator.save_dataset(data, reference_texts)
    
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(data)}")
    
    category_counts = {}
    for item in data:
        category = item["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    
    for category, count in category_counts.items():
        print(f"{category}: {count} samples")


if __name__ == "__main__":
    main()
